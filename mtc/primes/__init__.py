"""
Number-Theoretic Geometric Computing: Residue Number Systems (RNS), Balanced Ternary (F3), and p-Adic Geometry.
"""

from .rns import PrimeResidueSystem
from .ternary import (
    pack_5_trits, unpack_5_trits,
    pack_ternary_vector, unpack_ternary_vector,
    ternary_multiplier_free_gemm
)
from .padic import (
    p_adic_valuation, p_adic_norm, p_adic_distance,
    tree_lca_depth, tree_ultrametric_distance
)

__all__ = [
    "PrimeResidueSystem",
    "pack_5_trits",
    "unpack_5_trits",
    "pack_ternary_vector",
    "unpack_ternary_vector",
    "ternary_multiplier_free_gemm",
    "p_adic_valuation",
    "p_adic_norm",
    "p_adic_distance",
    "tree_lca_depth",
    "tree_ultrametric_distance",
]
