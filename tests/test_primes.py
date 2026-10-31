"""
Unit tests for Number-Theoretic Computing: RNS, Balanced Ternary, and p-Adic Geometry.
"""

import numpy as np
import pytest
from mtc.primes.rns import PrimeResidueSystem
from mtc.primes.ternary import (
    pack_5_trits, unpack_5_trits,
    pack_ternary_vector, unpack_ternary_vector,
    ternary_multiplier_free_gemm
)
from mtc.primes.padic import (
    p_adic_valuation, p_adic_norm, p_adic_distance,
    tree_ultrametric_distance
)


def test_rns_crt_roundtrip():
    rns = PrimeResidueSystem(primes=[3, 5, 7, 11, 13])
    # M = 3 * 5 * 7 * 11 * 13 = 15015
    # Test across dynamic range [-7000, 7000]
    test_vals = np.array([-7000, -1024, -42, 0, 1, 42, 256, 1337, 7000], dtype=np.int64)
    residues = rns.to_residues(test_vals)
    reconstructed = rns.from_residues(residues, signed=True)
    np.testing.assert_array_equal(test_vals, reconstructed)


def test_rns_parallel_dot_product_carry_free():
    rns = PrimeResidueSystem(primes=[3, 5, 7, 11, 13, 17])
    # Compute vector dot product: <A, B>
    A = np.array([12, -7, 15, 3, -8, 22], dtype=np.int64)
    B = np.array([-4, 11, 2, -9, 5, 1], dtype=np.int64)

    expected_dot = int(np.dot(A, B))  # Integer result: -12*4 - 77 + 30 - 27 - 40 + 22 = -140

    A_res = rns.to_residues(A)
    B_res = rns.to_residues(B)

    # Compute dot product strictly in residue channels (zero carry between channels!)
    dot_res = rns.parallel_dot_product(A_res, B_res)
    recon_dot = int(rns.from_residues(dot_res, signed=True))

    assert recon_dot == expected_dot


def test_rns_gemm():
    rns = PrimeResidueSystem(primes=[3, 5, 7, 11, 13, 17, 19])
    A = np.random.randint(-15, 15, size=(4, 5)).astype(np.int64)
    B = np.random.randint(-15, 15, size=(5, 3)).astype(np.int64)

    expected_C = A @ B

    A_res = rns.to_residues(A)
    B_res = rns.to_residues(B)

    C_res = rns.rns_gemm(A_res, B_res)
    recon_C = rns.from_residues(C_res, signed=True)

    np.testing.assert_array_equal(recon_C, expected_C)


def test_ternary_5_trits_packing_lossless():
    # Test all 3^5 = 243 possible 5-trit combinations
    for t0 in [-1, 0, 1]:
        for t1 in [-1, 0, 1]:
            for t2 in [-1, 0, 1]:
                for t3 in [-1, 0, 1]:
                    for t4 in [-1, 0, 1]:
                        byte_val = pack_5_trits(t0, t1, t2, t3, t4)
                        assert 0 <= byte_val <= 242
                        u0, u1, u2, u3, u4 = unpack_5_trits(byte_val)
                        assert (u0, u1, u2, u3, u4) == (t0, t1, t2, t3, t4)


def test_ternary_vector_packing():
    np.random.seed(42)
    length = 73  # Non-multiple of 5
    trits = np.random.choice(np.array([-1, 0, 1], dtype=np.int8), size=length)
    packed = pack_ternary_vector(trits)
    # 73 trits fit in ceil(73/5) = 15 bytes! (vs 19 bytes with 2-bit packing)
    assert len(packed) == 15
    unpacked = unpack_ternary_vector(packed, length)
    np.testing.assert_array_equal(trits, unpacked)


def test_ternary_multiplier_free_gemm():
    np.random.seed(42)
    M, K, N = 16, 32, 8
    W = np.random.choice(np.array([-1, 0, 1], dtype=np.int8), size=(M, K))
    X = np.random.randn(K, N).astype(np.float32)

    expected_Y = W.astype(np.float32) @ X

    actual_Y = np.empty((M, N), dtype=np.float32)
    ternary_multiplier_free_gemm(W, X, actual_Y)

    np.testing.assert_allclose(actual_Y, expected_Y, atol=1e-5)


def test_padic_valuation_and_ultrametric():
    # 3-adic valuation of 18: 18 = 2 * 3^2 -> v_3(18) = 2
    assert p_adic_valuation(18, 3) == 2
    assert p_adic_norm(18, 3) == 1.0 / 9.0

    # 2-adic valuation of 40: 40 = 5 * 2^3 -> v_2(40) = 3
    assert p_adic_valuation(40, 2) == 3
    assert p_adic_norm(40, 2) == 1.0 / 8.0

    # Verify Strong Triangle Inequality for random integers:
    # d_p(x, z) <= max(d_p(x, y), d_p(y, z))
    p = 3
    np.random.seed(42)
    for _ in range(50):
        x = int(np.random.randint(-1000, 1000))
        y = int(np.random.randint(-1000, 1000))
        z = int(np.random.randint(-1000, 1000))

        d_xz = p_adic_distance(x, z, p)
        d_xy = p_adic_distance(x, y, p)
        d_yz = p_adic_distance(y, z, p)

        assert d_xz <= max(d_xy, d_yz) + 1e-12


def test_tree_ultrametric_distance():
    # Path A: [0, 1, 0] (depth 3)
    # Path B: [0, 1, 1] (depth 3, LCA depth = 2)
    # Path C: [1, 0, 0] (depth 3, LCA depth with A is 0)
    path_a = [0, 1, 0]
    path_b = [0, 1, 1]
    path_c = [1, 0, 0]

    d_ab = tree_ultrametric_distance(path_a, path_b, base=2.0)  # 2^(-2) = 0.25
    d_bc = tree_ultrametric_distance(path_b, path_c, base=2.0)  # 2^0 = 1.0
    d_ac = tree_ultrametric_distance(path_a, path_c, base=2.0)  # 2^0 = 1.0

    assert d_ab == 0.25
    assert d_ac == 1.0
    # Strong triangle inequality: d(A, C) <= max(d(A, B), d(B, C)) -> 1.0 <= max(0.25, 1.0) = 1.0
    assert d_ac <= max(d_ab, d_bc)


if __name__ == "__main__":
    test_rns_crt_roundtrip()
    test_rns_parallel_dot_product_carry_free()
    test_rns_gemm()
    test_ternary_5_trits_packing_lossless()
    test_ternary_vector_packing()
    test_ternary_multiplier_free_gemm()
    test_padic_valuation_and_ultrametric()
    test_tree_ultrametric_distance()
    print("All Primes & Number-Theoretic tests passed!")
