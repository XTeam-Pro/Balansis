"""ACT-compensated Singular Value Decomposition for Balansis.

Implements SVD using Golub-Kahan bidiagonalization followed by
implicit QR iteration on the bidiagonal matrix. ACT-compensated
inner products and norms are used at critical numerical points
to maintain stability. Falls back to numpy SVD if ACT diverges.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

from balansis.core.absolute import AbsoluteValue
from balansis.core.operations import Operations

# Type alias for matrix of AbsoluteValues
Matrix = List[List[AbsoluteValue]]

# Algorithm constants
_MAX_QR_ITERATIONS = 500
_CONVERGENCE_TOL = 1e-14
_FALLBACK_MULTIPLIER = 100.0


@dataclass
class CompensatedSVDResult:
    """Result of ACT-compensated SVD.

    Attributes:
        U: Left singular vectors (m x k matrix of AbsoluteValue).
        S: Singular values as list of AbsoluteValue (length k).
        Vt: Right singular vectors transposed (k x n matrix).
        compensation_factors: Compensation factors from each step.
        reconstruction_error: Frobenius norm ||A - U diag(S) Vt||.
    """

    U: Matrix
    S: List[AbsoluteValue]
    Vt: Matrix
    compensation_factors: List[float]
    reconstruction_error: float

    def __iter__(self):
        """Support tuple unpacking: U, S, Vt = svd(a)."""
        return iter((self.U, self.S, self.Vt))

    def __len__(self) -> int:
        """Length for backward compatibility with (U, S, Vt) tuple."""
        return 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numpy_to_matrix(arr: np.ndarray) -> Matrix:
    """Convert numpy array to Matrix of AbsoluteValues."""
    m, n = arr.shape
    return [
        [AbsoluteValue.from_float(float(arr[i, j])) for j in range(n)]
        for i in range(m)
    ]


def _matrix_to_numpy(mat: Matrix) -> np.ndarray:
    """Convert Matrix of AbsoluteValues to numpy array."""
    return np.array(
        [[v.to_float() for v in row] for row in mat], dtype=float
    )


def _compensated_inner_product(
    x: np.ndarray, y: np.ndarray,
) -> Tuple[float, float]:
    """ACT-compensated inner product via Operations.sequence_sum."""
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
    """ACT-compensated 2-norm."""
    dot_val, comp = _compensated_inner_product(x, x)
    if dot_val < 0:
        dot_val = 0.0
    return math.sqrt(dot_val), comp


def _householder_vector(x: np.ndarray) -> Tuple[np.ndarray, float, List[float]]:
    """Compute Householder vector v and tau = 2/(v^T v).

    Args:
        x: Input vector.

    Returns:
        (v, tau, compensations) where H = I - tau v v^T maps
        x to -sign(x_0) ||x|| e_1.
    """
    comps: List[float] = []
    norm_x, comp = _compensated_norm(x)
    comps.append(comp)

    if norm_x < 1e-15:
        return x.copy(), 0.0, comps

    sign_x0 = 1.0 if x[0] >= 0 else -1.0
    v = x.copy()
    v[0] += sign_x0 * norm_x

    vtv, vtv_comp = _compensated_inner_product(v, v)
    comps.append(vtv_comp)

    tau = 2.0 / vtv if abs(vtv) > 1e-30 else 0.0
    return v, tau, comps


def _givens_rotation(a: float, b: float) -> Tuple[float, float, float]:
    """Compute Givens rotation (c, s, r) with ACT-compensated norm.

    Returns (c, s, r) such that::

        [ c  s] [a]   [r]
        [-s  c] [b] = [0]
    """
    if abs(b) < 1e-30:
        return 1.0, 0.0, a
    if abs(a) < 1e-30:
        return 0.0, math.copysign(1.0, b), abs(b)
    vec = np.array([a, b])
    r, _ = _compensated_norm(vec)
    if r < 1e-30:
        return 1.0, 0.0, 0.0
    c = a / r
    s = b / r
    return c, s, r


# ---------------------------------------------------------------------------
# Phase 1: Bidiagonalization
# ---------------------------------------------------------------------------

def _bidiagonalize(
    A_np: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """Reduce A to upper bidiagonal form B = U^T A V.

    Uses Householder reflections from left (each column) and right
    (each row) with ACT-compensated inner products.

    Args:
        A_np: Input matrix of shape (m, n) with m >= n.

    Returns:
        (U, B, V, compensations) where U is m x m, B is m x n
        (upper bidiagonal in top n x n block), and V is n x n.
    """
    m, n = A_np.shape
    B = A_np.copy().astype(float)
    U = np.eye(m, dtype=float)
    V = np.eye(n, dtype=float)
    comps: List[float] = []

    for j in range(n):
        # Left Householder: zero out B[j+1:, j]
        if j < m:
            x = B[j:, j].copy()
            v, tau, h_comps = _householder_vector(x)
            comps.extend(h_comps)

            if tau > 0:
                for col in range(j, n):
                    vt_col, comp = _compensated_inner_product(v, B[j:, col])
                    comps.append(comp)
                    B[j:, col] -= tau * vt_col * v

                for row_idx in range(m):
                    uv, comp = _compensated_inner_product(
                        U[row_idx, j:], v,
                    )
                    comps.append(comp)
                    U[row_idx, j:] -= tau * uv * v

        # Right Householder: zero out B[j, j+2:]
        if j < n - 2:
            x = B[j, j + 1:].copy()
            v, tau, h_comps = _householder_vector(x)
            comps.extend(h_comps)

            if tau > 0:
                for row_idx in range(j, m):
                    bv, comp = _compensated_inner_product(
                        B[row_idx, j + 1:], v,
                    )
                    comps.append(comp)
                    B[row_idx, j + 1:] -= tau * bv * v

                for row_idx in range(n):
                    vv, comp = _compensated_inner_product(
                        V[row_idx, j + 1:], v,
                    )
                    comps.append(comp)
                    V[row_idx, j + 1:] -= tau * vv * v

    return U, B, V, comps


# ---------------------------------------------------------------------------
# Phase 2: Bidiagonal SVD via implicit QR
# ---------------------------------------------------------------------------

def _golub_kahan_step(
    B: np.ndarray,
    U_b: np.ndarray,
    Vt_b: np.ndarray,
    p: int,
    q: int,
) -> List[float]:
    """One Golub-Kahan SVD step on the active block B[p:q+1, p:q+1].

    Computes a Wilkinson shift from the trailing 2x2 of B^T B and
    chases the resulting bulge using Givens rotations.

    Args:
        B: Dense bidiagonal matrix (modified in place).
        U_b: Left rotation accumulator (modified in place).
        Vt_b: Right rotation accumulator (modified in place).
        p: Start index of active block.
        q: End index of active block.

    Returns:
        List of compensation factors from Givens rotations.
    """
    comps: List[float] = []
    n_full = B.shape[0]

    # Wilkinson shift from trailing 2x2 of T = B^T B
    d_qm1 = B[q - 1, q - 1]
    d_q = B[q, q]
    f_qm1 = B[q - 1, q]
    f_qm2 = B[q - 2, q - 1] if q - 1 > p else 0.0

    a11 = d_qm1 * d_qm1 + f_qm2 * f_qm2
    a12 = d_qm1 * f_qm1
    a22 = d_q * d_q + f_qm1 * f_qm1

    delta = (a11 - a22) / 2.0
    if abs(delta) < 1e-30:
        mu = a22 - abs(a12)
    else:
        sgn = 1.0 if delta >= 0 else -1.0
        mu = a22 - a12 * a12 / (
            delta + sgn * math.sqrt(delta * delta + a12 * a12)
        )

    # Initialise bulge chase
    y = B[p, p] * B[p, p] - mu
    z = B[p, p] * B[p, p + 1]

    for k in range(p, q):
        # Right Givens to zero z
        c, s, _ = _givens_rotation(y, z)
        comps.append(0.0)

        # Apply to columns k, k+1 of B
        for i in range(max(0, k - 1), min(q + 2, n_full)):
            t1 = c * B[i, k] + s * B[i, k + 1]
            t2 = -s * B[i, k] + c * B[i, k + 1]
            B[i, k] = t1
            B[i, k + 1] = t2

        # Update Vt_b
        for col in range(Vt_b.shape[1]):
            t1 = c * Vt_b[k, col] + s * Vt_b[k + 1, col]
            t2 = -s * Vt_b[k, col] + c * Vt_b[k + 1, col]
            Vt_b[k, col] = t1
            Vt_b[k + 1, col] = t2

        # Left Givens to zero B[k+1, k] (the bulge)
        y_l = B[k, k]
        z_l = B[k + 1, k]
        c, s, _ = _givens_rotation(y_l, z_l)
        comps.append(0.0)

        # Apply to rows k, k+1 of B
        for j_col in range(k, min(q + 2, n_full)):
            t1 = c * B[k, j_col] + s * B[k + 1, j_col]
            t2 = -s * B[k, j_col] + c * B[k + 1, j_col]
            B[k, j_col] = t1
            B[k + 1, j_col] = t2

        # Update U_b
        for row in range(U_b.shape[0]):
            t1 = c * U_b[row, k] + s * U_b[row, k + 1]
            t2 = -s * U_b[row, k] + c * U_b[row, k + 1]
            U_b[row, k] = t1
            U_b[row, k + 1] = t2

        # Setup next iteration
        if k < q - 1:
            y = B[k, k + 1]
            z = B[k, k + 2]

    return comps


def _bidiag_qr_svd(
    B_sq: np.ndarray,
    n: int,
    max_iter: int = _MAX_QR_ITERATIONS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """SVD of n x n bidiagonal matrix via implicit QR iteration.

    Args:
        B_sq: Upper bidiagonal matrix (n x n), copied internally.
        n: Matrix size.
        max_iter: Maximum number of QR iterations.

    Returns:
        (B_diag, U_b, Vt_b, compensations) where B_diag is
        approximately diagonal and B_sq = U_b B_diag Vt_b.
    """
    B = B_sq.copy()
    U_b = np.eye(n, dtype=float)
    Vt_b = np.eye(n, dtype=float)
    comps: List[float] = []

    for iteration in range(max_iter):
        # Global convergence check
        converged = True
        for i in range(n - 1):
            if abs(B[i, i + 1]) > _CONVERGENCE_TOL * (
                abs(B[i, i]) + abs(B[i + 1, i + 1])
            ):
                converged = False
                break
        if converged:
            break

        # Find largest unreduced block [p, q]
        q = n - 1
        while q > 0 and abs(B[q - 1, q]) <= _CONVERGENCE_TOL * (
            abs(B[q - 1, q - 1]) + abs(B[q, q])
        ):
            q -= 1
        if q == 0:
            break

        p = q - 1
        while p > 0 and abs(B[p - 1, p]) > _CONVERGENCE_TOL * (
            abs(B[p - 1, p - 1]) + abs(B[p, p])
        ):
            p -= 1

        # Deflation: check for zero diagonal in [p, q]
        zero_found = False
        for i in range(p, q):
            if abs(B[i, i]) < _CONVERGENCE_TOL:
                # Zero out superdiagonal using left Givens rotations
                for j_sweep in range(i, q):
                    if abs(B[i, j_sweep + 1]) < 1e-30:
                        break
                    c, s, _ = _givens_rotation(
                        B[j_sweep + 1, j_sweep + 1], B[i, j_sweep + 1],
                    )
                    for col in range(n):
                        t1 = c * B[j_sweep + 1, col] + s * B[i, col]
                        t2 = -s * B[j_sweep + 1, col] + c * B[i, col]
                        B[j_sweep + 1, col] = t1
                        B[i, col] = t2
                    for row in range(n):
                        t1 = c * U_b[row, j_sweep + 1] + s * U_b[row, i]
                        t2 = -s * U_b[row, j_sweep + 1] + c * U_b[row, i]
                        U_b[row, j_sweep + 1] = t1
                        U_b[row, i] = t2
                zero_found = True
                break
        if zero_found:
            continue

        # Standard Golub-Kahan step
        step_comps = _golub_kahan_step(B, U_b, Vt_b, p, q)
        comps.extend(step_comps)

    return B, U_b, Vt_b, comps


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def _svd_numpy_fallback(A_np: np.ndarray) -> CompensatedSVDResult:
    """Compute SVD using numpy as a fallback.

    Args:
        A_np: Input matrix.

    Returns:
        CompensatedSVDResult wrapping numpy results.
    """
    U_np, S_np, Vt_np = np.linalg.svd(A_np, full_matrices=False)
    recon = U_np @ np.diag(S_np) @ Vt_np
    recon_err = float(np.linalg.norm(A_np - recon, "fro"))

    U_mat = _numpy_to_matrix(U_np)
    S_abs = [AbsoluteValue.from_float(float(s)) for s in S_np]
    Vt_mat = _numpy_to_matrix(Vt_np)

    return CompensatedSVDResult(
        U=U_mat,
        S=S_abs,
        Vt=Vt_mat,
        compensation_factors=[1.0],
        reconstruction_error=recon_err,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def svd(
    a: Matrix,
    method: str = "householder",
) -> CompensatedSVDResult:
    """ACT-compensated Singular Value Decomposition.

    Computes the thin SVD  A = U diag(S) Vt  using Golub-Kahan
    bidiagonalization followed by implicit QR iteration. All inner
    products and norms in the Householder reflections use ACT-compensated
    arithmetic. If the compensated algorithm diverges, falls back to
    numpy SVD gracefully.

    Args:
        a: Input matrix as list of lists of AbsoluteValue.
        method: Currently only ``"householder"`` is supported.

    Returns:
        CompensatedSVDResult containing U, S, Vt, compensation_factors,
        and reconstruction_error. Supports tuple unpacking for backward
        compatibility: ``U, S, Vt = svd(a)``.

    Raises:
        ValueError: If matrix is empty or method is unknown.
    """
    if not a or not a[0]:
        raise ValueError("Matrix must be non-empty")
    if method != "householder":
        raise ValueError(
            f"Unknown SVD method: {method!r}. Only 'householder' is supported."
        )

    A_np = _matrix_to_numpy(a)
    m, n = A_np.shape

    try:
        result = _svd_act(A_np)

        # Validate: compare reconstruction error with numpy
        np_U, np_S, np_Vt = np.linalg.svd(A_np, full_matrices=False)
        np_recon_err = float(
            np.linalg.norm(A_np - np_U @ np.diag(np_S) @ np_Vt, "fro")
        )
        threshold = _FALLBACK_MULTIPLIER * max(np_recon_err, 1e-10)

        if result.reconstruction_error > threshold:
            warnings.warn(
                "ACT SVD diverged (reconstruction error "
                f"{result.reconstruction_error:.2e} > threshold "
                f"{threshold:.2e}), falling back to numpy SVD",
                stacklevel=2,
            )
            return _svd_numpy_fallback(A_np)

        return result

    except Exception:
        warnings.warn(
            "ACT SVD encountered an error, falling back to numpy SVD",
            stacklevel=2,
        )
        return _svd_numpy_fallback(A_np)


def _svd_act(A_np: np.ndarray) -> CompensatedSVDResult:
    """Full ACT-compensated SVD implementation.

    Args:
        A_np: Input matrix of shape (m, n).

    Returns:
        CompensatedSVDResult.
    """
    m, n = A_np.shape
    transposed = m < n
    if transposed:
        A_np = A_np.T.copy()
        m, n = n, m

    compensation_factors: List[float] = []

    # Phase 1: Bidiagonalize
    U_h, B, V_h, bidiag_comps = _bidiagonalize(A_np)
    compensation_factors.extend(bidiag_comps)

    # Phase 2: SVD of the n x n bidiagonal block
    B_sq = B[:n, :n].copy()
    B_diag, U_b, Vt_b, qr_comps = _bidiag_qr_svd(B_sq, n)
    compensation_factors.extend(qr_comps)

    # Extract singular values (diagonal of B_diag)
    S_raw = np.array([B_diag[i, i] for i in range(n)])

    # Make singular values non-negative, adjust U
    for i in range(n):
        if S_raw[i] < 0:
            S_raw[i] = -S_raw[i]
            U_b[:, i] = -U_b[:, i]

    # Sort in descending order
    idx = np.argsort(-S_raw)
    S_sorted = S_raw[idx]
    U_b = U_b[:, idx]
    Vt_b = Vt_b[idx, :]

    # Combine transformations
    # A = U_h B V_h^T, B[:n,:n] = U_b diag(S) Vt_b
    # A = U_h[:,: n] U_b diag(S) Vt_b V_h^T
    U_final = U_h[:, :n] @ U_b  # m x n
    Vt_final = Vt_b @ V_h.T     # n x n

    # Handle transpose case
    if transposed:
        # SVD(A^T) = U_f S Vt_f  =>  SVD(A) = Vt_f^T S U_f^T
        U_out = Vt_final.T
        Vt_out = U_final.T
        A_original = A_np.T
    else:
        U_out = U_final
        Vt_out = Vt_final
        A_original = A_np

    # Reconstruction error
    recon = U_out @ np.diag(S_sorted) @ Vt_out
    recon_err = float(np.linalg.norm(A_original - recon, "fro"))

    # Convert to AbsoluteValue types
    U_mat = _numpy_to_matrix(U_out)
    S_abs = [AbsoluteValue.from_float(float(s)) for s in S_sorted]
    Vt_mat = _numpy_to_matrix(Vt_out)

    return CompensatedSVDResult(
        U=U_mat,
        S=S_abs,
        Vt=Vt_mat,
        compensation_factors=compensation_factors,
        reconstruction_error=recon_err,
    )
