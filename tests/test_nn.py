"""
Unit tests for Neural Network Layers and Models (FP32, INT8, MTC-Adaptive, and Balanced Ternary).
"""

import numpy as np
import pytest

from mtc.nn.layers import LinearFP32, LinearINT8, LinearMTCAdaptive, LinearTernaryBitNet
from mtc.nn.model import DeepMLP_FP32, DeepMLP_INT8, DeepMLP_MTCAdaptive, DeepMLP_Ternary


def test_linear_layers_forward():
    X = np.random.randn(8, 32).astype(np.float32)

    l_fp32 = LinearFP32(32, 16, seed=42)
    l_int8 = LinearINT8(32, 16, seed=42)
    l_mtc = LinearMTCAdaptive(32, 16, block_size=16, seed=42)
    l_ternary = LinearTernaryBitNet(32, 16, seed=42)

    y_fp32 = l_fp32.forward(X)
    y_int8 = l_int8.forward(X)
    y_mtc = l_mtc.forward(X)
    y_ternary = l_ternary.forward(X)

    assert y_fp32.shape == (8, 16)
    assert y_int8.shape == (8, 16)
    assert y_mtc.shape == (8, 16)
    assert y_ternary.shape == (8, 16)

    # Check that MTC-Adaptive matches FP32 within quantization bound
    rel_err_mtc = np.linalg.norm(y_fp32 - y_mtc) / np.linalg.norm(y_fp32)
    assert rel_err_mtc < 0.05


def test_ternary_backward_ste():
    X = np.random.randn(4, 16).astype(np.float32)
    layer = LinearTernaryBitNet(16, 8, seed=42)

    y = layer.forward(X)
    grad_out = np.ones_like(y, dtype=np.float32)

    grad_x, grad_w, grad_b = layer.backward(grad_out)
    assert grad_x.shape == (4, 16)
    assert grad_w.shape == (16, 8)
    assert grad_b.shape == (8,)

    # Test step
    W_latent_before = layer.W_latent.copy()
    layer.step(lr=0.01, grad_w=grad_w, grad_b=grad_b)
    assert not np.array_equal(layer.W_latent, W_latent_before)


def test_deep_mlp_inference_all():
    X = np.random.randn(4, 64).astype(np.float32)

    m_fp32 = DeepMLP_FP32(64, 128, 16, seed=42)
    m_int8 = DeepMLP_INT8(64, 128, 16, seed=42)
    m_mtc = DeepMLP_MTCAdaptive(64, 128, 16, seed=42)
    m_ternary = DeepMLP_Ternary(64, 128, 16, seed=42)

    out_fp32, _ = m_fp32.forward(X)
    out_int8 = m_int8.forward(X)
    out_mtc = m_mtc.forward(X)
    out_ternary, _ = m_ternary.forward(X)

    assert out_fp32.shape == (4, 16)
    assert out_int8.shape == (4, 16)
    assert out_mtc.shape == (4, 16)
    assert out_ternary.shape == (4, 16)


if __name__ == "__main__":
    test_linear_layers_forward()
    test_ternary_backward_ste()
    test_deep_mlp_inference_all()
    print("All Neural Network tests passed!")
