"""
Unit tests for Stage 3: Adaptive Precision and Joint Precision Negotiation.
"""

import numpy as np
import pytest
from mtc.core.types import PrecisionMode
from mtc.core.precision import (
    quantize_and_dequantize_block,
    compute_relative_error,
    select_independent_precision,
    negotiate_joint_precision
)


def test_independent_precision_selection():
    np.random.seed(42)
    # Block that fits well in P2: discrete or near-ternary
    block_p2 = np.array([[1.0, -1.0], [0.0, 1.0]], dtype=np.float32)
    p_sel, _ = select_independent_precision(block_p2, epsilon=0.01)
    assert p_sel == PrecisionMode.P2

    # Continuous gaussian block: P2 will have high error, P4 or P8 needed
    block_gauss = np.random.randn(4, 4).astype(np.float32)
    p_sel_strict, _ = select_independent_precision(block_gauss, epsilon=0.02)
    # Should select P8 or higher for strict 2% error
    assert p_sel_strict in (PrecisionMode.P8, PrecisionMode.P16)


def test_joint_precision_negotiation():
    # If block A has tiny magnitude, precision on B can be relaxed
    block_a = np.array([[0.001, 0.002], [0.001, 0.003]], dtype=np.float32)
    block_b = np.array([[5.0, -3.0], [2.0, 4.0]], dtype=np.float32)

    pa, pb, _, _ = negotiate_joint_precision(block_a, block_b, target_epsilon=0.05, reduction_tiles=4)
    # Because norm_a is tiny, error impact on C is minimal; minimal bits selected
    assert pa.value + pb.value <= 16

    # If both blocks require high precision:
    block_a_rnd = np.random.uniform(-10, 10, size=(4, 4)).astype(np.float32)
    block_b_rnd = np.random.uniform(-10, 10, size=(4, 4)).astype(np.float32)

    pa_strict, pb_strict, _, _ = negotiate_joint_precision(
        block_a_rnd, block_b_rnd, target_epsilon=0.01, reduction_tiles=1
    )
    # For strict 1% product error, higher precision is negotiated
    assert pa_strict.value >= 8 or pb_strict.value >= 8


if __name__ == "__main__":
    test_independent_precision_selection()
    test_joint_precision_negotiation()
    print("All Stage 3 precision tests passed!")
