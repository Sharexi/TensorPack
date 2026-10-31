"""
Multi-Layer Neural Network Architectures for FP32, INT8, MTC-8, and Balanced Ternary.
"""

from typing import List, Tuple
import numpy as np

from .layers import LinearFP32, LinearINT8, LinearMTCAdaptive, LinearTernaryBitNet


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_backward(grad_out: np.ndarray, x: np.ndarray) -> np.ndarray:
    return grad_out * (x > 0.0)


class DeepMLP_FP32:
    """Multi-layer FP32 Reference Neural Network."""

    def __init__(self, in_dim: int = 256, hidden_dim: int = 512, out_dim: int = 16, seed: int = 42):
        self.fc1 = LinearFP32(in_dim, hidden_dim, seed=seed)
        self.fc2 = LinearFP32(hidden_dim, hidden_dim, seed=seed + 1)
        self.fc3 = LinearFP32(hidden_dim, in_dim, seed=seed + 2)
        self.head = LinearFP32(in_dim, out_dim, seed=seed + 3)
        self.layers = [self.fc1, self.fc2, self.fc3, self.head]

    @property
    def total_bytes(self) -> int:
        return sum(l.weight_bytes for l in self.layers)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        # Forward pass returning output and intermediate activations for backprop
        h1 = self.fc1.forward(x)
        a1 = relu(h1)

        h2 = self.fc2.forward(a1)
        a2 = relu(h2)

        h3 = self.fc3.forward(a2)
        a3 = relu(h3)

        out = self.head.forward(a3)
        return out, [x, h1, a1, h2, a2, h3, a3]

    def backward_and_step(self, grad_loss: np.ndarray, cache: List[np.ndarray], lr: float = 0.01):
        x, h1, a1, h2, a2, h3, a3 = cache

        # Head backward
        gh3, gw_head, gb_head = self.head.backward(grad_loss)
        self.head.step(lr, gw_head, gb_head)

        # FC3 backward
        ga2 = relu_backward(gh3, h3)
        gh2, gw3, gb3 = self.fc3.backward(ga2)
        self.fc3.step(lr, gw3, gb3)

        # FC2 backward
        ga1 = relu_backward(gh2, h2)
        gh1, gw2, gb2 = self.fc2.backward(ga1)
        self.fc2.step(lr, gw2, gb2)

        # FC1 backward
        gx = relu_backward(gh1, h1)
        _, gw1, gb1 = self.fc1.backward(gx)
        self.fc1.step(lr, gw1, gb1)


class DeepMLP_INT8:
    """Multi-layer Conventional INT8 Quantized Model."""

    def __init__(self, in_dim: int = 256, hidden_dim: int = 512, out_dim: int = 16, seed: int = 42):
        self.fc1 = LinearINT8(in_dim, hidden_dim, seed=seed)
        self.fc2 = LinearINT8(hidden_dim, hidden_dim, seed=seed + 1)
        self.fc3 = LinearINT8(hidden_dim, in_dim, seed=seed + 2)
        self.head = LinearINT8(in_dim, out_dim, seed=seed + 3)
        self.layers = [self.fc1, self.fc2, self.fc3, self.head]

    @property
    def total_bytes(self) -> int:
        return sum(l.weight_bytes for l in self.layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        a1 = relu(self.fc1.forward(x))
        a2 = relu(self.fc2.forward(a1))
        a3 = relu(self.fc3.forward(a2))
        return self.head.forward(a3)


class DeepMLP_MTCAdaptive:
    """Multi-layer MTC-8 Adaptive Pre-Packed Model."""

    def __init__(self, in_dim: int = 256, hidden_dim: int = 512, out_dim: int = 16, seed: int = 42):
        self.fc1 = LinearMTCAdaptive(in_dim, hidden_dim, seed=seed)
        self.fc2 = LinearMTCAdaptive(hidden_dim, hidden_dim, seed=seed + 1)
        self.fc3 = LinearMTCAdaptive(hidden_dim, in_dim, seed=seed + 2)
        self.head = LinearMTCAdaptive(in_dim, out_dim, seed=seed + 3)
        self.layers = [self.fc1, self.fc2, self.fc3, self.head]

    @property
    def total_bytes(self) -> int:
        return sum(l.weight_bytes for l in self.layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        a1 = relu(self.fc1.forward(x))
        a2 = relu(self.fc2.forward(a1))
        a3 = relu(self.fc3.forward(a2))
        return self.head.forward(a3)


class DeepMLP_Ternary:
    """Multi-layer Balanced Ternary (BitNet-style) Model with Multiplier-Free Compute & Training."""

    def __init__(self, in_dim: int = 256, hidden_dim: int = 512, out_dim: int = 16, seed: int = 42):
        self.fc1 = LinearTernaryBitNet(in_dim, hidden_dim, seed=seed)
        self.fc2 = LinearTernaryBitNet(hidden_dim, hidden_dim, seed=seed + 1)
        self.fc3 = LinearTernaryBitNet(hidden_dim, in_dim, seed=seed + 2)
        self.head = LinearTernaryBitNet(in_dim, out_dim, seed=seed + 3)
        self.layers = [self.fc1, self.fc2, self.fc3, self.head]

    @property
    def total_bytes(self) -> int:
        return sum(l.weight_bytes for l in self.layers)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        h1 = self.fc1.forward(x)
        a1 = relu(h1)

        h2 = self.fc2.forward(a1)
        a2 = relu(h2)

        h3 = self.fc3.forward(a2)
        a3 = relu(h3)

        out = self.head.forward(a3)
        return out, [x, h1, a1, h2, a2, h3, a3]

    def backward_and_step(self, grad_loss: np.ndarray, cache: List[np.ndarray], lr: float = 0.01):
        x, h1, a1, h2, a2, h3, a3 = cache

        # Head backward
        gh3, gw_head, gb_head = self.head.backward(grad_loss)
        self.head.step(lr, gw_head, gb_head)

        # FC3 backward (multiplier-free)
        ga2 = relu_backward(gh3, h3)
        gh2, gw3, gb3 = self.fc3.backward(ga2)
        self.fc3.step(lr, gw3, gb3)

        # FC2 backward (multiplier-free)
        ga1 = relu_backward(gh2, h2)
        gh1, gw2, gb2 = self.fc2.backward(ga1)
        self.fc2.step(lr, gw2, gb2)

        # FC1 backward (multiplier-free)
        gx = relu_backward(gh1, h1)
        _, gw1, gb1 = self.fc1.backward(gx)
        self.fc1.step(lr, gw1, gb1)
