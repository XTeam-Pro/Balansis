"""Tests for ACT-compensated QR decomposition."""

import pytest
import numpy as np

from balansis.core.absolute import AbsoluteValue
from balansis.linalg.qr import qr_decompose, CompensatedQRResult
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


class TestQRHouseholder:
    """Tests for Householder QR decomposition."""

    def test_identity_matrix(self):
        """QR of identity should give Q=I, R=I."""
        I = np.eye(3)
        a = _make_matrix(I)
        result = qr_decompose(a, method="householder")
        Q = _to_numpy(result.Q)
        R = _to_numpy(result.R)
        assert np.allclose(np.abs(Q), np.eye(3), atol=1e-10)
        assert np.allclose(np.abs(R), np.eye(3), atol=1e-10)

    def test_orthogonality(self):
        """Q^T Q should be close to identity."""
        np.random.seed(42)
        A = np.random.randn(5, 3)
        a = _make_matrix(A)
        result = qr_decompose(a, method="householder")
        Q = _to_numpy(result.Q)
        QtQ = Q.T @ Q
        assert np.allclose(QtQ, np.eye(3), atol=1e-10)

    def test_orthogonality_error_tracked(self):
        """orthogonality_error should be small."""
        np.random.seed(42)
        A = np.random.randn(4, 3)
        a = _make_matrix(A)
        result = qr_decompose(a, method="householder")
        assert result.orthogonality_error < 1e-10

    def test_upper_triangularity(self):
        """R should be upper triangular."""
        np.random.seed(42)
        A = np.random.randn(4, 3)
        a = _make_matrix(A)
        result = qr_decompose(a, method="householder")
        R = _to_numpy(result.R)
        for i in range(R.shape[0]):
            for j in range(i):
                assert abs(R[i, j]) < 1e-10, f"R[{i},{j}] = {R[i, j]} is not zero"

    def test_reconstruction(self):
        """A should equal Q @ R."""
        np.random.seed(42)
        A = np.random.randn(5, 3)
        a = _make_matrix(A)
        result = qr_decompose(a, method="householder")
        Q = _to_numpy(result.Q)
        R = _to_numpy(result.R)
        recon = Q @ R
        assert np.allclose(A, recon, atol=1e-10)

    def test_square_matrix(self):
        """QR of a square matrix."""
        np.random.seed(123)
        A = np.random.randn(4, 4)
        a = _make_matrix(A)
        result = qr_decompose(a, method="householder")
        Q = _to_numpy(result.Q)
        R = _to_numpy(result.R)
        assert np.allclose(A, Q @ R, atol=1e-10)
        assert np.allclose(Q.T @ Q, np.eye(4), atol=1e-10)

    def test_ill_conditioned(self):
        """QR of an ill-conditioned matrix should still produce orthogonal Q."""
        A = np.array([
            [1.0, 1.0, 1.0],
            [1e-10, 1e-10, 0.0],
            [1e-10, 0.0, 1e-10],
            [0.0, 1e-10, 1e-10],
        ])
        a = _make_matrix(A)
        result = qr_decompose(a, method="householder")
        Q = _to_numpy(result.Q)
        assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-8)
        R = _to_numpy(result.R)
        assert np.allclose(A, Q @ R, atol=1e-8)

    def test_tall_matrix(self):
        """QR of a tall matrix (m >> n)."""
        np.random.seed(99)
        A = np.random.randn(10, 2)
        a = _make_matrix(A)
        result = qr_decompose(a, method="householder")
        Q = _to_numpy(result.Q)
        R = _to_numpy(result.R)
        assert Q.shape == (10, 2)
        assert R.shape == (2, 2)
        assert np.allclose(A, Q @ R, atol=1e-10)

    def test_single_column(self):
        """QR of a single-column matrix."""
        A = np.array([[3.0], [4.0]])
        a = _make_matrix(A)
        result = qr_decompose(a, method="householder")
        Q = _to_numpy(result.Q)
        R = _to_numpy(result.R)
        assert Q.shape == (2, 1)
        assert R.shape == (1, 1)
        assert np.allclose(A, Q @ R, atol=1e-10)

    def test_compensation_factors_populated(self):
        """compensation_factors should be a non-empty list."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        a = _make_matrix(A)
        result = qr_decompose(a, method="householder")
        assert isinstance(result.compensation_factors, list)
        assert len(result.compensation_factors) > 0


class TestQRGivens:
    """Tests for Givens rotation QR decomposition."""

    def test_reconstruction(self):
        """A should equal Q @ R."""
        np.random.seed(42)
        A = np.random.randn(4, 3)
        a = _make_matrix(A)
        result = qr_decompose(a, method="givens")
        Q = _to_numpy(result.Q)
        R = _to_numpy(result.R)
        assert np.allclose(A, Q @ R, atol=1e-10)

    def test_orthogonality(self):
        """Q^T Q should be identity."""
        np.random.seed(42)
        A = np.random.randn(4, 3)
        a = _make_matrix(A)
        result = qr_decompose(a, method="givens")
        Q = _to_numpy(result.Q)
        assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-10)

    def test_upper_triangularity(self):
        """R should be upper triangular."""
        np.random.seed(42)
        A = np.random.randn(4, 3)
        a = _make_matrix(A)
        result = qr_decompose(a, method="givens")
        R = _to_numpy(result.R)
        for i in range(R.shape[0]):
            for j in range(i):
                assert abs(R[i, j]) < 1e-10

    def test_identity(self):
        """QR of identity matrix."""
        I = np.eye(3)
        a = _make_matrix(I)
        result = qr_decompose(a, method="givens")
        Q = _to_numpy(result.Q)
        R = _to_numpy(result.R)
        assert np.allclose(np.abs(Q), np.eye(3), atol=1e-10)
        assert np.allclose(np.abs(R), np.eye(3), atol=1e-10)


class TestQRGramSchmidt:
    """Tests for Modified Gram-Schmidt QR with reorthogonalization."""

    def test_reconstruction(self):
        """A should equal Q @ R."""
        np.random.seed(42)
        A = np.random.randn(4, 3)
        a = _make_matrix(A)
        result = qr_decompose(a, method="gram_schmidt")
        Q = _to_numpy(result.Q)
        R = _to_numpy(result.R)
        assert np.allclose(A, Q @ R, atol=1e-10)

    def test_orthogonality(self):
        """Q^T Q should be identity."""
        np.random.seed(42)
        A = np.random.randn(5, 3)
        a = _make_matrix(A)
        result = qr_decompose(a, method="gram_schmidt")
        Q = _to_numpy(result.Q)
        assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-10)

    def test_reorthogonalization_improves_accuracy(self):
        """MGS with reorthogonalization should have low orthogonality error."""
        np.random.seed(42)
        A = np.random.randn(6, 4)
        a = _make_matrix(A)
        result = qr_decompose(a, method="gram_schmidt")
        assert result.orthogonality_error < 1e-10

    def test_ill_conditioned(self):
        """MGS with reorthogonalization on ill-conditioned matrix."""
        A = np.array([
            [1.0, 1.0],
            [1e-4, 0.0],
            [0.0, 1e-4],
        ])
        a = _make_matrix(A)
        result = qr_decompose(a, method="gram_schmidt")
        Q = _to_numpy(result.Q)
        R = _to_numpy(result.R)
        assert np.allclose(A, Q @ R, atol=1e-8)
        assert np.allclose(Q.T @ Q, np.eye(2), atol=1e-8)


class TestQREdgeCases:
    """Edge case tests for QR decomposition."""

    def test_empty_matrix_raises(self):
        """Empty matrix should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            qr_decompose([])

    def test_unknown_method_raises(self):
        """Unknown method should raise ValueError."""
        A = np.eye(2)
        a = _make_matrix(A)
        with pytest.raises(ValueError, match="Unknown QR method"):
            qr_decompose(a, method="unknown")

    def test_returns_compensated_result(self):
        """Should return CompensatedQRResult."""
        A = np.eye(2)
        a = _make_matrix(A)
        result = qr_decompose(a)
        assert isinstance(result, CompensatedQRResult)

    def test_backward_compat_unpacking(self):
        """Tuple unpacking Q, R = qr_decompose(a) should work."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        a = _make_matrix(A)
        Q, R = qr_decompose(a)
        assert len(Q) == 2
        assert len(R) == 2

    def test_1x1_matrix(self):
        """QR of a 1x1 matrix."""
        A = np.array([[5.0]])
        a = _make_matrix(A)
        result = qr_decompose(a)
        Q = _to_numpy(result.Q)
        R = _to_numpy(result.R)
        assert np.allclose(A, Q @ R, atol=1e-10)

    def test_diagonal_matrix(self):
        """QR of a diagonal matrix."""
        A = np.diag([3.0, 4.0, 5.0])
        a = _make_matrix(A)
        result = qr_decompose(a)
        Q = _to_numpy(result.Q)
        R = _to_numpy(result.R)
        assert np.allclose(A, Q @ R, atol=1e-10)

    def test_all_methods_agree(self):
        """All three methods should give consistent Q @ R = A."""
        np.random.seed(42)
        A = np.random.randn(4, 3)
        a = _make_matrix(A)
        for method in ["householder", "givens", "gram_schmidt"]:
            result = qr_decompose(a, method=method)
            Q = _to_numpy(result.Q)
            R = _to_numpy(result.R)
            assert np.allclose(A, Q @ R, atol=1e-9), f"Failed for {method}"

    def test_negative_values(self):
        """QR with negative entries."""
        A = np.array([[-1.0, 2.0], [3.0, -4.0], [-5.0, 6.0]])
        a = _make_matrix(A)
        result = qr_decompose(a)
        Q = _to_numpy(result.Q)
        R = _to_numpy(result.R)
        assert np.allclose(A, Q @ R, atol=1e-10)

    def test_comparison_with_numpy(self):
        """Singular values of Q^T Q should be all 1s, matching numpy behaviour."""
        np.random.seed(42)
        A = np.random.randn(5, 3)
        a = _make_matrix(A)
        result = qr_decompose(a)
        Q = _to_numpy(result.Q)
        sv = np.linalg.svd(Q, compute_uv=False)
        assert np.allclose(sv, np.ones(3), atol=1e-10)
