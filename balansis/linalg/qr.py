# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.
#
# This file is part of Balansis.
# Balansis is dual-licensed under:
#   1. GNU Affero General Public License v3.0 (AGPLv3) for open-source use.
#   2. A Commercial License for proprietary and corporate use.
#
# See LICENSING.md in the project root for license selection details.
# For commercial licensing: andrew@xteam.pro
"""ACT-compensated QR decomposition.

Three numerical methods are supported via the ``method=`` argument:
- ``"householder"`` (default): numerically stable Householder reflections,
  recommended for general dense matrices.
- ``"givens"``: Givens rotations, suitable for sparse or banded structures.
- ``"gram_schmidt"``: Modified Gram-Schmidt with re-orthogonalization for
  improved accuracy on ill-conditioned inputs.

The returned :class:`CompensatedQRResult` wraps Q, R, an orthogonality
diagnostic, and the per-step compensation factors. For backward
compatibility, the result is iterable so ``Q, R = qr_decompose(A)`` still
works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Tuple

import numpy as np

from balansis.core.absolute import AbsoluteValue

Matrix = List[List[AbsoluteValue]]


@dataclass
class CompensatedQRResult:
    """Result of an ACT-compensated QR decomposition.

    Attributes:
        Q: Orthogonal factor of shape (m, k) where k = min(m, n).
        R: Upper-triangular factor of shape (k, n).
        orthogonality_error: Frobenius norm of ``QᵀQ - I``; lower is better.
        method: Algorithm used (``"householder"``, ``"givens"`` or
            ``"gram_schmidt"``).
        compensation_factors: Per-iteration compensation factors emitted by
            the underlying numeric kernel; useful for diagnostics.
    """

    Q: Matrix
    R: Matrix
    orthogonality_error: float
    method: str
    compensation_factors: List[float] = field(default_factory=list)

    def __iter__(self) -> Iterator[Matrix]:
        # Backwards-compatible tuple unpacking: Q, R = qr_decompose(a)
        yield self.Q
        yield self.R


def _to_numpy(mat: Matrix) -> np.ndarray:
    return np.array([[v.to_float() for v in row] for row in mat], dtype=np.float64)


def _from_numpy(arr: np.ndarray) -> Matrix:
    return [
        [AbsoluteValue.from_float(float(arr[i, j])) for j in range(arr.shape[1])]
        for i in range(arr.shape[0])
    ]


def _householder_qr(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    m, n = A.shape
    k = min(m, n)
    R = A.copy()
    Q = np.eye(m, dtype=np.float64)
    compensations: List[float] = []
    for j in range(k):
        x = R[j:, j]
        norm_x = float(np.linalg.norm(x))
        if norm_x == 0.0:
            compensations.append(1.0)
            continue
        sign = -1.0 if x[0] >= 0 else 1.0  # Choose reflection direction for stability
        v = x.copy()
        v[0] -= sign * norm_x
        v_norm_sq = float(np.dot(v, v))
        if v_norm_sq == 0.0:
            compensations.append(1.0)
            continue
        beta = 2.0 / v_norm_sq
        # Apply H = I - beta v vᵀ to the trailing submatrix and Q
        R[j:, j:] -= beta * np.outer(v, v @ R[j:, j:])
        Q[:, j:] -= beta * np.outer(Q[:, j:] @ v, v)
        compensations.append(1.0 + abs(beta))
    R_out = R[:k, :n]
    Q_out = Q[:, :k]
    return Q_out, R_out, compensations


def _givens_qr(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    m, n = A.shape
    k = min(m, n)
    R = A.copy()
    Q = np.eye(m, dtype=np.float64)
    compensations: List[float] = []
    for j in range(k):
        for i in range(m - 1, j, -1):
            a, b = R[i - 1, j], R[i, j]
            if b == 0.0:
                continue
            r = float(np.hypot(a, b))
            c = a / r
            s = -b / r
            # Apply rotation to rows i-1 and i
            row_top = c * R[i - 1, :] - s * R[i, :]
            row_bot = s * R[i - 1, :] + c * R[i, :]
            R[i - 1, :] = row_top
            R[i, :] = row_bot
            # Apply same rotation (transposed) to Q's columns
            col_left = c * Q[:, i - 1] - s * Q[:, i]
            col_right = s * Q[:, i - 1] + c * Q[:, i]
            Q[:, i - 1] = col_left
            Q[:, i] = col_right
            compensations.append(1.0 + abs(s))
    if not compensations:
        compensations.append(1.0)
    return Q[:, :k], R[:k, :n], compensations


def _modified_gram_schmidt_qr(
    A: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    m, n = A.shape
    k = min(m, n)
    Q = np.zeros((m, k), dtype=np.float64)
    R = np.zeros((k, n), dtype=np.float64)
    compensations: List[float] = []
    V = A[:, :k].copy()
    for j in range(k):
        # First sweep
        norm = float(np.linalg.norm(V[:, j]))
        if norm == 0.0:
            R[j, j] = 0.0
            continue
        R[j, j] = norm
        Q[:, j] = V[:, j] / norm
        compensations.append(1.0)
        for i in range(j + 1, k):
            R[j, i] = float(Q[:, j] @ V[:, i])
            V[:, i] -= R[j, i] * Q[:, j]
        # Re-orthogonalization (second sweep) — improves accuracy on ill-conditioned input
        for i in range(j + 1, k):
            r2 = float(Q[:, j] @ V[:, i])
            V[:, i] -= r2 * Q[:, j]
            R[j, i] += r2
    if n > k:
        # If A has more columns than rows, project remaining columns
        for j in range(k, n):
            for i in range(k):
                R[i, j] = float(Q[:, i] @ A[:, j])
    return Q, R, compensations


def qr_decompose(
    a: Matrix,
    method: str = "householder",
) -> CompensatedQRResult:
    """ACT-compensated QR decomposition.

    Args:
        a: Input matrix as list of rows of :class:`AbsoluteValue`. Must be
            non-empty.
        method: One of ``"householder"``, ``"givens"``, or ``"gram_schmidt"``.

    Returns:
        :class:`CompensatedQRResult` with ``Q``, ``R``, an
        ``orthogonality_error`` score, and per-step compensation factors.

    Raises:
        ValueError: If ``a`` is empty or ``method`` is not recognised.
    """
    if not a or (a and not a[0]):
        raise ValueError("Cannot decompose an empty / non-empty matrix")
    A = _to_numpy(a)

    method_lc = method.lower()
    if method_lc == "householder":
        Q_np, R_np, compensations = _householder_qr(A)
    elif method_lc == "givens":
        Q_np, R_np, compensations = _givens_qr(A)
    elif method_lc == "gram_schmidt":
        Q_np, R_np, compensations = _modified_gram_schmidt_qr(A)
    else:
        raise ValueError(f"Unknown QR method: {method}")

    k = Q_np.shape[1]
    orthogonality_error = float(np.linalg.norm(Q_np.T @ Q_np - np.eye(k)))

    return CompensatedQRResult(
        Q=_from_numpy(Q_np),
        R=_from_numpy(R_np),
        orthogonality_error=orthogonality_error,
        method=method_lc,
        compensation_factors=compensations,
    )
