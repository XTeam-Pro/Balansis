# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.
#
# This file is part of Balansis.
# Balansis is dual-licensed under:
#   1. GNU Affero General Public License v3.0 (AGPLv3) for open-source use.
#   2. A Commercial License for proprietary and corporate use.
#
# See LICENSING.md in the project root for license selection details.
# For commercial licensing: andrew@xteam.pro
"""ACT-compensated Singular Value Decomposition.

``svd`` returns a :class:`CompensatedSVDResult` carrying U, S, Vᵀ together
with reconstruction error and per-singular-value compensation factors. The
underlying numerical kernel delegates to NumPy's ``np.linalg.svd`` (LAPACK
``gesdd``) for robustness on general dense matrices; the ACT layer adds the
diagnostic envelope and the AbsoluteValue lifting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np

from balansis.core.absolute import AbsoluteValue
from balansis.core.eternity import SingularArithmeticEvent, SingularPolicy
from balansis.core.operations import Operations
from balansis.core._eft import dot2

_SVD_METHODS = ("numpy_gesdd", "act_jacobi")


def _act_jacobi_svd(
    A: np.ndarray, tol: float = 1e-15, max_sweeps: int = 60
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-sided Jacobi SVD with ACT-compensated inner products.

    All column inner products (the Gram entries that drive each rotation and the
    final singular values) are computed with the correctly-rounded
    :func:`balansis.array.gram_pair` and :func:`balansis.core._eft.dot2`,
    so the decomposition genuinely uses ACT
    compensated arithmetic rather than delegating to LAPACK. One-sided Jacobi is
    chosen because it attains high *relative* accuracy on the singular values of
    ill-conditioned matrices, where its accuracy benefits directly from the
    compensated dot product.

    Returns ``(U, S, Vt)`` with ``S`` sorted descending, analogous to
    ``np.linalg.svd(A, full_matrices=False)``.
    """
    from balansis.array import gram_pair

    # Columns stay contiguous throughout Jacobi sweeps and reach C without copies.
    W = np.array(A, dtype=np.float64, order="F", copy=True)
    m, n = W.shape
    transposed = False
    if m < n:
        W = W.T.copy(order="F")
        m, n = W.shape
        transposed = True

    V = np.eye(n, dtype=np.float64)
    for _ in range(max_sweeps):
        max_off = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                ci = W[:, i]
                cj = W[:, j]
                aii, ajj, aij = gram_pair(ci, cj)
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

Matrix = List[List[AbsoluteValue]]
Vector = List[AbsoluteValue]


@dataclass
class CompensatedSVDResult:
    """Result of an ACT-compensated SVD.

    Attributes:
        U: Left singular vectors of shape (m, k).
        S: Singular values of length k, sorted descending.
        Vt: Right singular vectors (transposed) of shape (k, n).
        reconstruction_error: ``||A - U·diag(S)·Vt||_F``.
        method: Algorithm used (``"numpy_gesdd"``).
        compensation_factors: Per-singular-value compensation factors.
    """

    U: Matrix
    S: Vector
    Vt: Matrix
    reconstruction_error: float
    method: str
    compensation_factors: List[float] = field(default_factory=list)
    singular_events: List[SingularArithmeticEvent] = field(default_factory=list)

    def singular_telemetry(self) -> List[dict[str, object]]:
        """Return machine-readable singular arithmetic telemetry for this decomposition."""
        return [event.model_dump(mode="json") for event in self.singular_events]

    def __iter__(self) -> Iterator[Union[Matrix, Vector]]:
        # Backwards-compatible tuple unpacking: U, S, Vt = svd(A)
        yield self.U
        yield self.S
        yield self.Vt


def _to_numpy(mat: Matrix) -> np.ndarray:
    return np.array([[v.to_float() for v in row] for row in mat], dtype=np.float64)


def _from_numpy(arr: np.ndarray) -> Matrix:
    return [[AbsoluteValue.from_float(float(arr[i, j])) for j in range(arr.shape[1])]
            for i in range(arr.shape[0])]


def svd(
    a: Matrix,
    method: str = "numpy_gesdd",
    singular_policy: SingularPolicy | str = SingularPolicy.PROPAGATE,
    saturation_limit: float = 1e12,
) -> CompensatedSVDResult:
    """ACT-compensated SVD.

    Args:
        a: Input matrix as list of rows of :class:`AbsoluteValue`. Must be
            non-empty.
        method: SVD backend; currently only ``"numpy_gesdd"`` is supported.

    Returns:
        :class:`CompensatedSVDResult` with U, S, Vᵀ, reconstruction error
        and per-singular-value compensation factors.

    Raises:
        ValueError: If ``a`` is empty or ``method`` is unknown.
    """
    if not a or (a and not a[0]):
        raise ValueError("Cannot decompose an empty / non-empty matrix")
    method_l = method.lower()
    if method_l not in _SVD_METHODS:
        raise ValueError(f"Unknown SVD method: {method}")

    A = _to_numpy(a)
    if method_l == "act_jacobi":
        U_np, S_np, Vt_np = _act_jacobi_svd(A)
    else:
        U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)

    # NumPy already returns singular values sorted in descending order.
    # Compensation per singular value: relative magnitude vs. the leading
    # singular value, in log10 scale (small singular values get higher comp).
    leading = float(S_np[0]) if S_np.size > 0 else 1.0
    compensations: List[float] = []
    singular_events: List[SingularArithmeticEvent] = []
    resolved_policy = SingularPolicy(singular_policy)
    leading_abs = AbsoluteValue.from_float(leading)
    for s in S_np:
        s_float = float(s)
        if leading == 0.0 or s_float == 0.0:
            compensations.append(1.0)
        else:
            compensations.append(1.0 + max(0.0, np.log10(leading / max(s_float, 1e-300))))

        _, _, event = Operations.compensated_divide_policy(
            leading_abs,
            AbsoluteValue.from_float(s_float),
            policy=resolved_policy,
            saturation_limit=saturation_limit,
        )
        if event is not None:
            singular_events.append(event)

    # Reconstruction error in original float space
    recon = U_np @ np.diag(S_np) @ Vt_np
    reconstruction_error = float(np.linalg.norm(A - recon))

    return CompensatedSVDResult(
        U=_from_numpy(U_np),
        S=[AbsoluteValue.from_float(float(s)) for s in S_np],
        Vt=_from_numpy(Vt_np),
        reconstruction_error=reconstruction_error,
        method=method.lower(),
        compensation_factors=compensations,
        singular_events=singular_events,
    )
