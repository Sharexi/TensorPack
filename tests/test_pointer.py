"""
Unit tests for Stage 4: Pointer hierarchy and DAG operations.
"""

import numpy as np
import pytest
from mtc.core.types import PrecisionMode, OpCode
from mtc.core.arena import MemoryArena
from mtc.core.microtensor import MicroTensor
from mtc.core.pointer_tensor import PointerTensor


def test_microtensor_leaf_roundtrip():
    arena = MemoryArena(1024 * 1024)
    # Test P2 2x2
    orig_p2 = np.array([[-1.0, 0.0], [1.0, -1.0]], dtype=np.float32)
    mt_p2 = MicroTensor.from_array(orig_p2, PrecisionMode.P2, arena)
    pt_p2 = PointerTensor.from_microtensor(mt_p2)
    np.testing.assert_allclose(pt_p2.to_array(), orig_p2, atol=1e-5)

    # Test P4 2x2
    orig_p4 = np.array([[-7.0, 3.0], [5.0, -2.0]], dtype=np.float32)
    mt_p4 = MicroTensor.from_array(orig_p4, PrecisionMode.P4, arena)
    pt_p4 = PointerTensor.from_microtensor(mt_p4)
    np.testing.assert_allclose(pt_p4.to_array(), orig_p4, atol=1e-5)


def test_recursive_grid_composition():
    arena = MemoryArena(1024 * 1024)
    # 4x4 matrix composed of 4 (2x2) microtensors
    M = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [-1.0, -2.0, -3.0, -4.0],
        [-5.0, -6.0, -7.0, -8.0]
    ], dtype=np.float32)

    pt = PointerTensor.from_matrix_recursive(M, leaf_shape=(2, 2), precision=PrecisionMode.P8, arena=arena)
    assert pt.shape == (4, 4)
    assert pt.is_composite
    d_nodes, o_nodes = pt.count_nodes()
    assert d_nodes == 4
    assert o_nodes == 0

    reconstructed = pt.to_array()
    np.testing.assert_allclose(reconstructed, M, atol=1e-1)


def test_pointer_tensor_dag_operations():
    arena = MemoryArena(1024 * 1024)
    A_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    B_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    A_pt = PointerTensor.from_microtensor(MicroTensor.from_array(A_np, PrecisionMode.P8, arena))
    B_pt = PointerTensor.from_microtensor(MicroTensor.from_array(B_np, PrecisionMode.P8, arena))

    # Matmul DAG
    C_pt = A_pt @ B_pt
    assert C_pt.is_op
    assert C_pt.op_code == OpCode.MATMUL
    d_nodes, o_nodes = C_pt.count_nodes()
    assert d_nodes == 2
    assert o_nodes == 1

    expected_mul = A_np @ B_np
    actual_mul = C_pt.evaluate()
    np.testing.assert_allclose(actual_mul, expected_mul, atol=0.2)

    # Add DAG
    D_pt = A_pt + B_pt
    expected_add = A_np + B_np
    np.testing.assert_allclose(D_pt.evaluate(), expected_add, atol=0.2)

    # Transpose
    T_pt = A_pt.T
    np.testing.assert_allclose(T_pt.to_array(), A_np.T, atol=1e-1)


if __name__ == "__main__":
    test_microtensor_leaf_roundtrip()
    test_recursive_grid_composition()
    test_pointer_tensor_dag_operations()
    print("All Stage 4 pointer tests passed!")
