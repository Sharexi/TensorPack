"""
GPTQ (Frantar et al., 2023): second-order error compensation for weight quantization.

Repository convention: Wt is (K, N) = (in_features, out_features), so GPTQ walks
over the K input channels (rows here; columns in the paper's W layout). For each
row k it rounds W[k] onto its grid, and spreads the rounding error over the
not-yet-quantized rows using the inverse Hessian H = X^T X of the layer input:

    err      = (w_k - q_k) / [H^-1]_kk
    W[k+1:] -= [H^-1]_{k, k+1:}^T  err          (lazy batched in blocks of 128)

Two grid sources share one implementation:

- group_int_grid: symmetric INT, fp16 absmax scale per group of g input rows and
  output column. This is exactly the RTN "group_int" baseline grid, so GPTQ vs
  RTN isolates the effect of error compensation. Bits: b + 16/g.
- fixed per-element grids (scale, lo, hi), e.g. the leaves of an MTC-QT-P
  allocation. The grid is decided first (by the RD allocator, from the original
  weights); GPTQ then only chooses the integer codes. The result is therefore a
  valid encoding of the same format at exactly the same bits.
"""

from typing import Optional, Tuple

import numpy as np
import torch


def _prepare_hinv(H: torch.Tensor, damp: float) -> Tuple[torch.Tensor, torch.Tensor]:
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    return torch.linalg.cholesky(Hinv, upper=True), dead


@torch.no_grad()
def gptq(
    Wt: torch.Tensor,
    H: torch.Tensor,
    grid: str = "fixed",
    S: Optional[torch.Tensor] = None,
    LO: Optional[torch.Tensor] = None,
    HI: Optional[torch.Tensor] = None,
    bits: int = 4,
    group: int = 128,
    act_order: bool = False,
    damp: float = 0.01,
    block: int = 128,
) -> torch.Tensor:
    """
    Returns the dequantized (K, N) matrix in the original row order.

    grid="fixed": per-element scale S and code range [LO, HI] (all (K, N)); S == 0 means P0 (dropped).
    grid="group": symmetric INT `bits`, fp16 scale per `group` rows. With act_order the
                  group scales are computed up front from the original weights (static
                  groups); otherwise lazily from the error-updated weights, as in GPTQ.
    """
    W = Wt.to(torch.float32).clone()
    K, N = W.shape
    H = H.to(torch.float32)
    qmax = 2 ** (bits - 1) - 1

    if grid == "group":
        assert K % group == 0 and block == group
        if act_order:  # static groups in the original order
            g = W.reshape(K // group, group, N).abs().amax(dim=1, keepdim=True) / qmax
            S = g.half().float().expand(K // group, group, N).reshape(K, N).clone()
            LO = torch.full_like(W, -qmax - 1)
            HI = torch.full_like(W, qmax)

    perm = torch.argsort(torch.diag(H), descending=True) if act_order else None
    if perm is not None:
        W, H = W[perm], H[perm][:, perm]
        if S is not None:
            S, LO, HI = S[perm], LO[perm], HI[perm]

    Hinv, dead = _prepare_hinv(H, damp)
    Q = torch.zeros_like(W)
    lazy_groups = grid == "group" and not act_order

    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1 = W[i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        if lazy_groups:  # i1 is a group boundary because block == group
            s = (W1.abs().amax(dim=0) / qmax).half().float()
            S1 = s.expand_as(W1)
            LO1 = torch.full_like(W1, -qmax - 1)
            HI1 = torch.full_like(W1, qmax)
        else:
            S1, LO1, HI1 = S[i1:i2], LO[i1:i2], HI[i1:i2]
        for i in range(i2 - i1):
            w = W1[i]
            s = S1[i]
            safe = torch.where(s == 0, torch.ones_like(s), s)
            q = torch.clamp(torch.round(w / safe), LO1[i], HI1[i]) * s
            Q[i1 + i] = q
            err = (w - q) / Hinv1[i, i]
            W1[i:] -= Hinv1[i, i:].unsqueeze(1) * err.unsqueeze(0)
            E1[i] = err
        W[i2:] -= Hinv[i1:i2, i2:].T @ E1

    if perm is not None:
        out = torch.empty_like(Q)
        out[perm] = Q
        Q = out
    return Q


def group_int_bits(bits: int, group: int = 128) -> float:
    """Same accounting as formats.group_int: fp16 scale per group."""
    return bits + 16.0 / group


def qt_element_grid(sizes, leaf, modes, scales, shape) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand a quadtree allocation into per-element (scale, lo, hi) arrays."""
    from mtc.core.allocation import QRANGE

    K, N = shape
    S = np.zeros((K, N), np.float32)
    LO = np.zeros((K, N), np.float32)
    HI = np.zeros((K, N), np.float32)
    lo_of = np.zeros(17, np.float32)
    hi_of = np.zeros(17, np.float32)
    for m, (lo, hi) in QRANGE.items():
        lo_of[m], hi_of[m] = lo, hi
    for s in sizes:
        lf = leaf[s]
        rep = lambda x: np.repeat(np.repeat(x, s, axis=0), s, axis=1)
        mask = rep(lf)
        S = np.where(mask, rep(np.where(lf, scales[s], 0.0)), S)
        LO = np.where(mask, rep(np.where(lf, lo_of[modes[s]], 0.0)), LO)
        HI = np.where(mask, rep(np.where(lf, hi_of[modes[s]], 0.0)), HI)
    return S, LO, HI
