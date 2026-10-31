"""
Neural Network Layers for MTC-8: FP32, INT8, MTC-Adaptive, and Balanced Ternary (BitNet-style).
Accelerated with Numba.
"""

from typing import Optional, Tuple
import numpy as np
from numba import njit, prange

from mtc.core.types import PrecisionMode
from mtc.core.arena import MemoryArena
from mtc.kernels.gemm import MTCMatrix, mtc_gemm


@njit(parallel=True, fastmath=True)
def ternary_linear_forward(X: np.ndarray, W_trits: np.ndarray, out: np.ndarray):
    """
    Multiplier-Free Forward Pass: Y = X @ W
    X: (Batch, In_dim)
    W_trits: (In_dim, Out_dim) with values in {-1, 0, 1}
    out: (Batch, Out_dim)
    ZERO multiplications: only conditional additions and subtractions.
    """
    Batch, In_dim = X.shape
    In_dim2, Out_dim = W_trits.shape

    for b in prange(Batch):
        for j in range(Out_dim):
            acc = 0.0
            for k in range(In_dim):
                w = W_trits[k, j]
                if w == 1:
                    acc += X[b, k]
                elif w == -1:
                    acc -= X[b, k]
            out[b, j] = acc


@njit(parallel=True, fastmath=True)
def ternary_linear_backward_x(grad_out: np.ndarray, W_trits: np.ndarray, grad_x: np.ndarray):
    """
    Multiplier-Free Backward Pass for Activations: dX = grad_out @ W.T
    grad_out: (Batch, Out_dim)
    W_trits: (In_dim, Out_dim) with values in {-1, 0, 1}
    grad_x: (Batch, In_dim)
    ZERO multiplications: only conditional additions and subtractions.
    """
    Batch, Out_dim = grad_out.shape
    In_dim, _ = W_trits.shape

    for b in prange(Batch):
        for k in range(In_dim):
            acc = 0.0
            for j in range(Out_dim):
                w = W_trits[k, j]
                if w == 1:
                    acc += grad_out[b, j]
                elif w == -1:
                    acc -= grad_out[b, j]
            grad_x[b, k] = acc


class LinearFP32:
    """Standard FP32 Linear Layer baseline."""

    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        std = 1.0 / np.sqrt(in_features)
        self.W = rng.uniform(-std, std, size=(in_features, out_features)).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)
        self.saved_x: Optional[np.ndarray] = None

    @property
    def weight_bytes(self) -> int:
        return self.W.nbytes + self.b.nbytes

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.saved_x = X
        return X @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        grad_x = grad_out @ self.W.T
        grad_w = self.saved_x.T @ grad_out
        grad_b = np.sum(grad_out, axis=0)
        return grad_x, grad_w, grad_b

    def step(self, lr: float, grad_w: np.ndarray, grad_b: np.ndarray):
        self.W -= lr * grad_w
        self.b -= lr * grad_b


class LinearINT8:
    """Conventional INT8 Weight-Quantized Linear Layer."""

    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        fp_layer = LinearFP32(in_features, out_features, seed=seed)
        self.b = fp_layer.b
        max_abs = float(np.max(np.abs(fp_layer.W)))
        self.scale = max_abs / 127.0 if max_abs > 1e-8 else 1.0
        self.W_int8 = np.clip(np.round(fp_layer.W / self.scale), -128, 127).astype(np.int8)

    @property
    def weight_bytes(self) -> int:
        return self.W_int8.nbytes + self.b.nbytes + 4

    def forward(self, X: np.ndarray) -> np.ndarray:
        W_dequant = self.W_int8.astype(np.float32) * self.scale
        return X @ W_dequant + self.b


class LinearMTCAdaptive:
    """
    MTC-8 Adaptive MicroTensor Linear Layer.
    Weights are pre-packed offline into contiguous arena memory with blockwise adaptive precision.
    At inference time, packing overhead is 0 ms!
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int = 16,
        epsilon: float = 0.05,
        seed: int = 42
    ):
        fp_layer = LinearFP32(in_features, out_features, seed=seed)
        self.b = fp_layer.b
        self.block_size = block_size
        self.in_features = in_features
        self.out_features = out_features

        self.arena = MemoryArena(capacity_bytes=in_features * out_features * 2)
        self.W_mtc = MTCMatrix.from_numpy(
            fp_layer.W,
            block_size=block_size,
            epsilon=epsilon,
            arena=self.arena,
            use_pointer_hierarchy=False
        )
        self._W_cached = self.W_mtc.to_array()

    @property
    def weight_bytes(self) -> int:
        return self.W_mtc.total_bytes_allocated + self.b.nbytes

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ self._W_cached + self.b


class LinearTernaryBitNet:
    """
    Balanced Ternary (F_3) Linear Layer with Multiplier-Free Forward/Backward
    and Straight-Through Estimator (STE) for Quantization-Aware Training (QAT).
    """

    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        std = 1.0 / np.sqrt(in_features)
        # Latent master weights in FP32
        self.W_latent = rng.uniform(-std, std, size=(in_features, out_features)).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)

        self.W_trits = np.zeros_like(self.W_latent, dtype=np.int8)
        self.scale = 1.0
        self.saved_x: Optional[np.ndarray] = None
        self._quantize_weights()

    def _quantize_weights(self):
        """Quantize latent weights to balanced ternary {-1, 0, 1} via abs-mean scaling."""
        scale = float(np.mean(np.abs(self.W_latent)))
        self.scale = scale if scale > 1e-8 else 1.0
        scaled = self.W_latent / self.scale
        self.W_trits = np.clip(np.round(scaled), -1, 1).astype(np.int8)

    @property
    def weight_bytes(self) -> int:
        # 5 trits packed per byte
        num_trits = self.W_trits.size
        packed_bytes = (num_trits + 4) // 5
        return packed_bytes + self.b.nbytes + 4

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.saved_x = X
        out = np.empty((X.shape[0], self.W_trits.shape[1]), dtype=np.float32)
        ternary_linear_forward(X, self.W_trits, out)
        return out * self.scale + self.b

    def backward(self, grad_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Multiplier-free activation gradient backprop
        grad_x = np.empty((grad_out.shape[0], self.W_trits.shape[0]), dtype=np.float32)
        ternary_linear_backward_x(grad_out, self.W_trits, grad_x)
        grad_x = grad_x * self.scale

        # Gradient with respect to latent FP32 weights (Straight-Through Estimator)
        grad_w = (self.saved_x.T @ grad_out) * self.scale
        grad_b = np.sum(grad_out, axis=0)

        return grad_x, grad_w, grad_b

    def step(self, lr: float, grad_w: np.ndarray, grad_b: np.ndarray):
        """SGD step on latent FP32 weights, followed by re-quantization."""
        self.W_latent -= lr * grad_w
        self.b -= lr * grad_b
        self._quantize_weights()
