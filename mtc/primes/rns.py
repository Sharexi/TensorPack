"""
Residue Number System (RNS): Carry-Free Parallel Arithmetic Across Prime Channels.
"""

from typing import List, Tuple
import numpy as np


class PrimeResidueSystem:
    """
    Residue Number System over a basis of pairwise coprime prime moduli.
    Enables carry-free parallel matrix and vector dot products across
    independent prime channels.
    """

    def __init__(self, primes: List[int] = [3, 5, 7, 11, 13]):
        self.primes = np.array(primes, dtype=np.int64)
        self.num_channels = len(primes)
        self.M = int(np.prod(self.primes))
        self.half_M = self.M // 2

        # Precompute CRT reconstruction constants
        # M_i = M // p_i
        self.M_i = np.array([self.M // int(p) for p in self.primes], dtype=np.int64)
        # y_i = (M_i)^(-1) mod p_i
        self.y_i = np.array([pow(int(mi), -1, int(p)) for mi, p in zip(self.M_i, self.primes)], dtype=np.int64)
        self.crt_weights = (self.M_i * self.y_i) % self.M

    def to_residues(self, x: np.ndarray) -> np.ndarray:
        """
        Convert integer array of shape (...) to residues of shape (..., num_channels).
        Supports signed integers centered around [-M//2, M//2].
        """
        arr = np.asarray(x, dtype=np.int64)
        # Shift negative values into [0, M)
        pos_arr = arr % self.M
        # Broadcast across prime channels
        return pos_arr[..., np.newaxis] % self.primes

    def from_residues(self, residues: np.ndarray, signed: bool = True) -> np.ndarray:
        """
        Reconstruct original integers from residues using the Chinese Remainder Theorem (CRT).
        residues has shape (..., num_channels).
        """
        res = np.asarray(residues, dtype=np.int64)
        # Sum over channels: (r_i * crt_weights_i) % M
        weighted = (res * self.crt_weights) % self.M
        x_recon = np.sum(weighted, axis=-1) % self.M

        if signed:
            # Map [0, M) back to [-half_M, half_M]
            x_recon = np.where(x_recon > self.half_M, x_recon - self.M, x_recon)

        return x_recon

    def parallel_dot_product(self, A_res: np.ndarray, B_res: np.ndarray) -> np.ndarray:
        """
        Compute dot product <A, B> strictly within independent prime residue channels.
        A_res: (K, num_channels)
        B_res: (K, num_channels)
        Returns: (num_channels,) residue vector with ZERO carry propagation between channels.
        """
        K, channels = A_res.shape
        assert channels == self.num_channels
        out_res = np.empty(channels, dtype=np.int64)

        for c in range(channels):
            p = self.primes[c]
            # Modular accumulation in channel c
            acc = 0
            for k in range(K):
                acc = (acc + int(A_res[k, c]) * int(B_res[k, c])) % p
            out_res[c] = acc

        return out_res

    def rns_gemm(self, A_res: np.ndarray, B_res: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication C = A @ B across RNS channels.
        A_res: (M, K, channels)
        B_res: (K, N, channels)
        Returns: C_res of shape (M, N, channels)
        """
        M_dim, K, ch = A_res.shape
        K2, N_dim, ch2 = B_res.shape
        assert K == K2 and ch == ch2

        C_res = np.empty((M_dim, N_dim, ch), dtype=np.int64)
        for c in range(ch):
            p = int(self.primes[c])
            # Pure integer modular GEMM in channel c
            a_c = A_res[:, :, c]
            b_c = B_res[:, :, c]
            C_res[:, :, c] = (a_c @ b_c) % p

        return C_res
