"""
p-Adic Non-Archimedean Geometry and Ultrametric Tree Distance.
"""

from typing import Tuple, List
import numpy as np


def p_adic_valuation(n: int, p: int) -> int:
    """
    Compute p-adic valuation v_p(n): the highest power of prime p dividing n.
    v_p(0) is formally infinity (represented as float('inf')).
    """
    if n == 0:
        return float('inf')
    n = abs(n)
    v = 0
    while n % p == 0:
        v += 1
        n //= p
    return v


def p_adic_norm(n: int, p: int) -> float:
    """
    Compute p-adic absolute value |n|_p = p^(-v_p(n)).
    |0|_p = 0.
    """
    if n == 0:
        return 0.0
    v = p_adic_valuation(n, p)
    return float(p ** (-v))


def p_adic_distance(x: int, y: int, p: int) -> float:
    """
    Compute p-adic distance d_p(x, y) = |x - y|_p.
    Satisfies the non-Archimedean strong triangle inequality:
        d_p(x, z) <= max(d_p(x, y), d_p(y, z))
    """
    return p_adic_norm(x - y, p)


def tree_lca_depth(path_a: List[int], path_b: List[int]) -> int:
    """
    Find depth of Lowest Common Ancestor (LCA) between two tree paths.
    """
    common = 0
    for a, b in zip(path_a, path_b):
        if a == b:
            common += 1
        else:
            break
    return common


def tree_ultrametric_distance(path_a: List[int], path_b: List[int], base: float = 2.0) -> float:
    """
    Compute the ultrametric distance between two tree leaf nodes:
    d(A, B) = base^(-LCA_depth(A, B)).
    Isomorphic to p-adic distance over branch indices.
    """
    if path_a == path_b:
        return 0.0
    lca = tree_lca_depth(path_a, path_b)
    return float(base ** (-lca))
