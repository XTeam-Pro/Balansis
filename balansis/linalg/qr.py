"""ACT-compensated QR decomposition for Balansis.

Implements QR decomposition using compensated arithmetic operations
to maintain numerical stability according to ACT principles.
Three methods are supported: Householder reflections (default),
Givens rotations, and Modified Gram-Schmidt with ACT reorthogonalization.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

from balansis.core.absolute import AbsoluteValue
from balansis.core.operations import Operations

# Type alias for matrix of AbsoluteValues
Matrix = List[List[AbsoluteValue]]


@dataclass
class CompensatedQRResult:
    """Result of ACT-compensated QR decomposition.

    Attributes:
        Q: Orthogonal matrix (m x k) where k = min(m, n).
        R: Upper triangular matrix (k x n).
        orthogonality_error: Frobenius norm ||Q^T Q - I||.
        compensation_factors: Compensation factors recorded at each step.
    """

    Q: Matrix
    R: Matrix
    orthogonality_error: float
    compensation_factors: List[float]

    def __iter__(self):
        """Support tuple unpacking for backward compatibility."""
        return iter((self.Q, self.R))

    def __len__(self) -> int:
        """Length for backward compatibility with (Q, R) tuple."""
        return 2


def _numpy_to_matrix(arr: np.ndarray) -> Matrix:
    """Convert a numpy 2D array to a Matrix of AbsoluteValues."""
    m, n = arr.shape
    return [
        [AbsoluteValue.from_float(float(arr[i, j])) for j in range(n)]
        for i in range(m)
    ]


def _matrix_to_numpy(mat: Matrix) -> np.ndarray:
    """Convert a Matrix of AbsoluteValues to a numpy array."""
    return np.array(
        [[v.to_float() for v in row] for row in mat], dtype=float
    )


def _compensated_inner_product(
    x: np.ndarray, y: np.ndarray,
) -> Tuple[float, float]:
    """Compute inner product using ACT-compensated Kahan summation.

    Converts element-wise products to AbsoluteValues, then uses
    Operations.sequence_sum for compensated accumulation.

    Args:
        x: First vector (1D numpy array).
        y: Second vector (1D numpy array).

    Returns:
        Tuple of (dot_product_value, compensation_error).
    """
    n = len(x)
    if n == 0:
        return 0.0, 0.0
    products: List[AbsoluteValue] = []
    for i in range(n):
        a_val = AbsoluteValue.from_float(float(x[i]))
        b_val = AbsoluteValue.from_float(float(y[i]))
        prod, _ = Operations.compensated_multiply(a_val, b_val)
        products.append(prod)
    result, comp_error = Operations.sequence_sum(products)
    return result.to_float(), comp_error


def _compensated_norm(x: np.ndarray) -> Tuple[float, float]:
    """Compute 2-norm using ACT-compensated inner product.

    Args:
        x: Vector (1D numpy array).

    Returns:
        Tuple of (norm_value, compensation_error).
    """
    dot_val, comp = _compensated_inner_product(x, x)
    if dot_val < 0:
        dot_val = 0.0
    return math.sqrt(dot_val), comp


def _orthogonality_error(Q: np.ndarray) -> float:
    """Compute ||Q^T Q - I||_F as orthogonality measure."""
    k = Q.shape[1]
    QtQ = Q.T @ Q
    return float(np.linalg.norm(QtQ - np.eye(k), "fro"))


def _qr_householder(A_np: np.ndarray) -> CompensatedQRResult:
    """QR via Householder reflections with ACT-compensated inner products.

    For each column j the Householder vector v is chosen so that
    H_j x = -sign(x_0)||x|| e_1 where x = A[j:, j]. All inner products
    and norms go through ACT compensated operations.

    Args:
        A_np: Input matrix of shape (m, n).

    Returns:
        CompensatedQRResult with thin Q (m x k) and R (k x n).
    """
    m, n = A_np.shape
    k = min(m, n)
    R = A_np.copy().astype(float)
    Q_acc = np.eye(m, dtype=float)
    compensation_factors: List[float] = []

    for j in range(k):
        x = R[j:, j].copy()
        norm_x, comp = _compensated_norm(x)
        compensation_factors.append(comp)

        if norm_x < 1e-15:
            continue

        sign_x0 = 1.0 if x[0] >= 0 else -1.0
        v = x.copy()
        v[0] += sign_x0 * norm_x

        vtv, vtv_comp = _compensated_inner_product(v, v)
        compensation_factors.append(vtv_comp)

        if abs(vtv) < 1e-30:
            continue

        tau = 2.0 / vtv

        # Update R: R[j:, col] -= tau * (v^T R[j:, col]) * v
        for col in range(j, n):
            vt_col, dot_comp = _compensated_inner_product(v, R[j:, col])
            compensation_factors.append(dot_comp)
            R[j:, col] -= tau * vt_col * v

        # Update Q_acc: Q_acc[:, j:] -= tau * (Q_acc[:, j:] v) v^T
        for row_idx in range(m):
            qv, dot_comp = _compensated_inner_product(Q_acc[row_idx, j:], v)
            compensation_factors.append(dot_comp)
            Q_acc[row_idx, j:] -= tau * qv * v

    orth_err = _orthogonality_error(Q_acc[:, :k])
    Q_mat = _numpy_to_matrix(Q_acc[:, :k])
    R_mat = _numpy_to_matrix(R[:k, :])

    return CompensatedQRResult(
        Q=Q_mat,
        R=R_mat,
        orthogonality_error=orth_err,
        compensation_factors=compensation_factors,
    )


def _qr_givens(A_np: np.ndarray) -> CompensatedQRResult:
    """QR via Givens rotations with ACT compensation.

    For each subdiagonal entry a Givens rotation is computed to zero
    it out. ACT compensation is applied when computing the rotation
    parameters c and s via compensated norm.

    Args:
        A_np: Input matrix of shape (m, n).

    Returns:
        CompensatedQRResult.
    """
    m, n = A_np.shape
    k = min(m, n)
    R = A_np.copy().astype(float)
    Q_acc = np.eye(m, dtype=float)
    compensation_factors: List[float] = []

    for j in range(k):
        for i in range(m - 1, j, -1):
            a_val = R[i - 1, j]
            b_val = R[i, j]
            if abs(b_val) < 1e-15:
                continue

            # Compute Givens rotation with ACT-compensated norm
            ab_vec = np.array([a_val, b_val])
            r_val, comp = _compensated_norm(ab_vec)
            compensation_factors.append(comp)
            c = a_val / r_val
            s = -b_val / r_val

            # Apply rotation to rows (i-1) and i of R
            for col in range(n):
                tmp1 = c * R[i - 1, col] - s * R[i, col]
                tmp2 = s * R[i - 1, col] + c * R[i, col]
                R[i - 1, col] = tmp1
                R[i, col] = tmp2

            # Apply rotation to columns (i-1) and i of Q_acc
            for row_idx in range(m):
                tmp1 = c * Q_acc[row_idx, i - 1] - s * Q_acc[row_idx, i]
                tmp2 = s * Q_acc[row_idx, i - 1] + c * Q_acc[row_idx, i]
                Q_acc[row_idx, i - 1] = tmp1
                Q_acc[row_idx, i] = tmp2

    orth_err = _orthogonality_error(Q_acc[:, :k])
    Q_mat = _numpy_to_matrix(Q_acc[:, :k])
    R_mat = _numpy_to_matrix(R[:k, :])

    return CompensatedQRResult(
        Q=Q_mat,
        R=R_mat,
        orthogonality_error=orth_err,
        compensation_factors=compensation_factors,
    )


def _qr_gram_schmidt(A_np: np.ndarray) -> CompensatedQRResult:
    """QR via Modified Gram-Schmidt with ACT reorthogonalization.

    Two-pass modified Gram-Schmidt: after the first orthogonalization
    pass a second reorthogonalization pass corrects for rounding errors.
    All inner products use ACT-compensated summation.

    Args:
        A_np: Input matrix of shape (m, n).

    Returns:
        CompensatedQRResult.
    """
    m, n = A_np.shape
    k = min(m, n)
    Q = np.zeros((m, k), dtype=float)
    R = np.zeros((k, n), dtype=float)
    compensation_factors: List[float] = []
    V = A_np[:, :k].copy().astype(float)

    for j in range(k):
        # First pass: orthogonalize against previous columns
        for i in range(j):
            rij, comp = _compensated_inner_product(Q[:, i], V[:, j])
            compensation_factors.append(comp)
            R[i, j] = rij
            V[:, j] -= rij * Q[:, i]

        # Reorthogonalization pass
        for i in range(j):
            s, comp = _compensated_inner_product(Q[:, i], V[:, j])
            compensation_factors.append(comp)
            R[i, j] += s
            V[:, j] -= s * Q[:, i]

        # Normalize
        norm_vj, comp = _compensated_norm(V[:, j])
        compensation_factors.append(comp)
        if norm_vj < 1e-15:
            Q[:, j] = 0.0
            R[j, j] = 0.0
        else:
            Q[:, j] = V[:, j] / norm_vj
            R[j, j] = norm_vj

    # Project remaining columns of A onto Q for fat matrices
    if n > k:
        for j in range(k, n):
            for i in range(k):
                rij, comp = _compensated_inner_product(Q[:, i], A_np[:, j])
                compensation_factors.append(comp)
                R[i, j] = rij

    orth_err = _orthogonality_error(Q)
    Q_mat = _numpy_to_matrix(Q)
    R_mat = _numpy_to_matrix(R)

    return CompensatedQRResult(
        Q=Q_mat,
        R=R_mat,
        orthogonality_error=orth_err,
        compensation_factors=compensation_factors,
    )


def qr_decompose(
    a: Matrix, method: str = "householder",
) -> CompensatedQRResult:
    """ACT-compensated QR decomposition.

    Decomposes matrix A into Q * R where Q is orthogonal and R is
    upper triangular. All critical inner products and norms use
    ACT-compensated arithmetic for improved numerical stability.

    Args:
        a: Input matrix as list of lists of AbsoluteValue.
        method: Decomposition method — ``"householder"`` (default),
            ``"givens"``, or ``"gram_schmidt"``.

    Returns:
        CompensatedQRResult containing Q, R, orthogonality_error,
        and compensation_factors. Supports tuple unpacking for
        backward compatibility: ``Q, R = qr_decompose(a)``.

    Raises:
        ValueError: If matrix is empty or method is unknown.
    """
    if not a or not a[0]:
        raise ValueError("Matrix must be non-empty")

    A_np = _matrix_to_numpy(a)

    if method == "householder":
        return _qr_householder(A_np)
    elif method == "givens":
        return _qr_givens(A_np)
    elif method == "gram_schmidt":
        return _qr_gram_schmidt(A_np)
    else:
        raise ValueError(
            f"Unknown QR method: {method!r}. "
            f"Choose from 'householder', 'givens', 'gram_schmidt'."
        )
