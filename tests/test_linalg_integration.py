"""Integration tests for Balansis linalg module.

Tests consistency between SVD, QR, and GEMM operations.
"""

import pytest
import numpy as np

from balansis.core.absolute import AbsoluteValue
from balansis.linalg.svd import svd
from balansis.linalg.qr import qr_decompose
from balansis.linalg.gemm import matmul
from balansis import ACT_EPSILON


def _make_matrix(arr):
    """Convert a numpy array to a Matrix of AbsoluteValues."""
    return [
        [AbsoluteValue.from_float(float(arr[i, j])) for j in range(arr.shape[1])]
        for i in range(arr.shape[0])
    ]


def _to_numpy(mat):
    """Convert a Matrix of AbsoluteValues to numpy array."""
    return np.array([[v.to_float() for v in row] for row in mat], dtype=float)


def _s_to_numpy(s_list):
    """Convert list of AbsoluteValue singular values to numpy array."""
    return np.array([v.to_float() for v in s_list], dtype=float)


class TestSVDQRConsistency:
    """Tests for consistency between SVD and QR decompositions."""

    def test_svd_of_orthogonal_q(self):
        """SVD of an orthogonal Q matrix should give singular values all 1."""
        np.random.seed(42)
        A = np.random.randn(4, 3)
        a = _make_matrix(A)
        qr_result = qr_decompose(a)
        Q = qr_result.Q

        svd_result = svd(Q)
        S = _s_to_numpy(svd_result.S)
        assert np.allclose(S, np.ones(3), atol=1e-6)

    def test_svd_singular_values_match_r_diagonal(self):
        """Singular values of A should relate to R from QR.

        For a full-rank matrix, the absolute values of R's diagonal
        should match the singular values if A = Q R and Q is orthogonal.
        This is only exact when R's off-diagonal is zero (diagonal A),
        but for general matrices R's diagonal magnitude correlates.
        """
        np.random.seed(42)
        A = np.random.randn(4, 3)
        a = _make_matrix(A)

        svd_result = svd(a)
        S_svd = _s_to_numpy(svd_result.S)

        qr_result = qr_decompose(a)
        R = _to_numpy(qr_result.R)
        R_diag = np.abs(np.diag(R))

        # Both should be in the same order of magnitude
        assert np.allclose(np.sort(S_svd), np.sort(R_diag), rtol=0.5)

    def test_qr_and_svd_reconstruction_agree(self):
        """Both QR and SVD should reconstruct the original matrix."""
        np.random.seed(42)
        A = np.random.randn(5, 3)
        a = _make_matrix(A)

        # QR reconstruction
        qr_result = qr_decompose(a)
        Q = _to_numpy(qr_result.Q)
        R = _to_numpy(qr_result.R)
        recon_qr = Q @ R

        # SVD reconstruction
        svd_result = svd(a)
        U = _to_numpy(svd_result.U)
        S = _s_to_numpy(svd_result.S)
        Vt = _to_numpy(svd_result.Vt)
        recon_svd = U @ np.diag(S) @ Vt

        assert np.allclose(recon_qr, recon_svd, atol=1e-8)
        assert np.allclose(A, recon_qr, atol=1e-8)
        assert np.allclose(A, recon_svd, atol=1e-8)


class TestGEMMSVDRoundtrip:
    """Tests for GEMM + SVD roundtrip consistency."""

    def test_svd_roundtrip_via_matmul(self):
        """U @ diag(S) @ Vt via matmul should reconstruct A."""
        np.random.seed(42)
        A = np.random.randn(3, 3)
        a = _make_matrix(A)

        result = svd(a)
        U = _to_numpy(result.U)
        S = _s_to_numpy(result.S)
        Vt = _to_numpy(result.Vt)

        # Build diag(S) as AbsoluteValue matrix
        k = len(S)
        S_mat = _make_matrix(np.diag(S))
        U_mat = result.U
        Vt_mat = result.Vt

        # U @ diag(S) via matmul
        US, _ = matmul(U_mat, S_mat)
        # (U @ diag(S)) @ Vt via matmul
        recon, _ = matmul(US, Vt_mat)

        recon_np = _to_numpy(recon)
        assert np.allclose(A, recon_np, atol=1e-6)

    def test_matmul_preserves_svd_structure(self):
        """matmul(U, Vt) for orthogonal U, Vt should give an orthogonal matrix."""
        np.random.seed(42)
        A = np.random.randn(3, 3)
        a = _make_matrix(A)

        result = svd(a)
        U_mat = result.U
        Vt_mat = result.Vt

        # U @ Vt via matmul
        product, _ = matmul(U_mat, Vt_mat)
        P = _to_numpy(product)

        # P should be an orthogonal matrix (since U and Vt are)
        PtP = P.T @ P
        assert np.allclose(PtP, np.eye(3), atol=1e-6)


class TestQRGEMMConsistency:
    """Tests for QR + GEMM consistency."""

    def test_qr_reconstruction_via_matmul(self):
        """Q @ R via matmul should reconstruct A."""
        np.random.seed(42)
        A = np.random.randn(4, 3)
        a = _make_matrix(A)

        qr_result = qr_decompose(a)
        Q_mat = qr_result.Q
        R_mat = qr_result.R

        recon, _ = matmul(Q_mat, R_mat)
        recon_np = _to_numpy(recon)
        assert np.allclose(A, recon_np, atol=1e-8)

    def test_qtq_via_matmul(self):
        """Q^T Q via matmul should give identity."""
        np.random.seed(42)
        A = np.random.randn(4, 3)
        a = _make_matrix(A)

        qr_result = qr_decompose(a)
        Q = _to_numpy(qr_result.Q)

        # Transpose Q
        Qt_np = Q.T
        Qt_mat = _make_matrix(Qt_np)
        Q_mat = qr_result.Q

        QtQ, _ = matmul(Qt_mat, Q_mat)
        QtQ_np = _to_numpy(QtQ)
        assert np.allclose(QtQ_np, np.eye(3), atol=1e-8)


class TestLinalgExportsIntegration:
    """Test that all linalg functions are importable and work together."""

    def test_all_imports(self):
        """All linalg functions should be importable."""
        from balansis.linalg import matmul, svd, qr_decompose
        assert callable(matmul)
        assert callable(svd)
        assert callable(qr_decompose)

    def test_full_pipeline(self):
        """Full pipeline: create matrix, QR, SVD, GEMM verify."""
        np.random.seed(42)
        A = np.random.randn(3, 3)
        a = _make_matrix(A)

        # QR decomposition
        Q, R = qr_decompose(a)
        Q_np = _to_numpy(Q)
        R_np = _to_numpy(R)
        assert np.allclose(A, Q_np @ R_np, atol=1e-8)

        # SVD
        U, S, Vt = svd(a)
        U_np = _to_numpy(U)
        S_np = _s_to_numpy(S)
        Vt_np = _to_numpy(Vt)
        assert np.allclose(A, U_np @ np.diag(S_np) @ Vt_np, atol=1e-8)

        # GEMM verify Q @ R
        recon, comp = matmul(Q, R)
        recon_np = _to_numpy(recon)
        assert np.allclose(A, recon_np, atol=1e-8)
