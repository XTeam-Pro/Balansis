# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro).
# AGPL-3.0-only OR commercial license; see LICENSING.md.
"""Frozen pre-fusion Jacobi SVD from d2f4e143, for regression/benchmarks only."""

import math
from typing import Tuple

import numpy as np

from balansis.core._eft import dot2


def _act_jacobi_svd(
    A: np.ndarray, tol: float = 1e-15, max_sweeps: int = 60
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-sided Jacobi SVD with ACT-compensated inner products.

    All column inner products (the Gram entries that drive each rotation and the
    final singular values) are computed with the correctly-rounded
    :func:`balansis.core._eft.dot2`, so the decomposition genuinely uses ACT
    compensated arithmetic rather than delegating to LAPACK. One-sided Jacobi is
    chosen because it attains high *relative* accuracy on the singular values of
    ill-conditioned matrices, where its accuracy benefits directly from the
    compensated dot product.

    Returns ``(U, S, Vt)`` with ``S`` sorted descending, analogous to
    ``np.linalg.svd(A, full_matrices=False)``.
    """
    W = np.asarray(A, dtype=np.float64).copy()
    m, n = W.shape
    transposed = False
    if m < n:
        W = W.T.copy()
        m, n = W.shape
        transposed = True

    V = np.eye(n, dtype=np.float64)
    for _ in range(max_sweeps):
        max_off = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                ci = W[:, i]
                cj = W[:, j]
                aii = dot2(ci, ci)
                ajj = dot2(cj, cj)
                aij = dot2(ci, cj)
                if aii <= 0.0 or ajj <= 0.0:
                    continue
                denom = math.sqrt(aii * ajj)
                if denom == 0.0:
                    continue
                rel = abs(aij) / denom
                if rel > max_off:
                    max_off = rel
                if rel <= tol:
                    continue
                # Jacobi rotation that diagonalizes [[aii, aij], [aij, ajj]].
                tau = (ajj - aii) / (2.0 * aij)
                t = math.copysign(1.0, tau) / (abs(tau) + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = c * t
                col_i = c * W[:, i] - s * W[:, j]
                col_j = s * W[:, i] + c * W[:, j]
                W[:, i] = col_i
                W[:, j] = col_j
                vi = c * V[:, i] - s * V[:, j]
                vj = s * V[:, i] + c * V[:, j]
                V[:, i] = vi
                V[:, j] = vj
        if max_off <= tol:
            break

    singular = np.array(
        [math.sqrt(max(dot2(W[:, k], W[:, k]), 0.0)) for k in range(n)],
        dtype=np.float64,
    )
    order = np.argsort(-singular)
    singular = singular[order]
    W = W[:, order]
    V = V[:, order]

    U = np.zeros((m, n), dtype=np.float64)
    for k in range(n):
        if singular[k] > 0.0:
            U[:, k] = W[:, k] / singular[k]

    Vt = V.T
    if transposed:
        # SVD(A^T) = U S Vt  =>  A = Vt^T S U^T
        U, Vt = Vt.T, U.T
    return U, singular, Vt
