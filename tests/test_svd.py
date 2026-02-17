"""Tests for ACT-compensated Singular Value Decomposition."""

import pytest
import numpy as np

from balansis.core.absolute import AbsoluteValue
from balansis.linalg.svd import svd, CompensatedSVDResult
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


class TestSVDBasic:
    """Basic SVD functionality tests."""

    def test_identity_matrix(self):
        """SVD of identity should give singular values all equal to 1."""
        I = np.eye(3)
        a = _make_matrix(I)
        result = svd(a)
        S = _s_to_numpy(result.S)
        assert np.allclose(S, np.ones(3), atol=1e-10)

    def test_diagonal_matrix(self):
        """SVD of diagonal matrix should return sorted singular values."""
        D = np.diag([5.0, 3.0, 1.0])
        a = _make_matrix(D)
        result = svd(a)
        S = _s_to_numpy(result.S)
        # Singular values should be sorted descending
        assert np.allclose(S, [5.0, 3.0, 1.0], atol=1e-10)

    def test_diagonal_unsorted(self):
        """SVD should sort singular values in descending order."""
        D = np.diag([1.0, 5.0, 3.0])
        a = _make_matrix(D)
        result = svd(a)
        S = _s_to_numpy(result.S)
        assert np.allclose(S, [5.0, 3.0, 1.0], atol=1e-10)

    def test_reconstruction(self):
        """A should equal U @ diag(S) @ Vt."""
        np.random.seed(42)
        A = np.random.randn(4, 3)
        a = _make_matrix(A)
        result = svd(a)
        U = _to_numpy(result.U)
        S = _s_to_numpy(result.S)
        Vt = _to_numpy(result.Vt)
        recon = U @ np.diag(S) @ Vt
        assert np.allclose(A, recon, atol=1e-8)

    def test_reconstruction_error_small(self):
        """Reconstruction error should be small."""
        np.random.seed(42)
        A = np.random.randn(4, 3)
        a = _make_matrix(A)
        result = svd(a)
        assert result.reconstruction_error < 1e-8

    def test_singular_values_non_negative(self):
        """All singular values should be non-negative."""
        np.random.seed(42)
        A = np.random.randn(5, 3)
        a = _make_matrix(A)
        result = svd(a)
        S = _s_to_numpy(result.S)
        assert np.all(S >= 0)

    def test_singular_values_descending(self):
        """Singular values should be in descending order."""
        np.random.seed(42)
        A = np.random.randn(5, 4)
        a = _make_matrix(A)
        result = svd(a)
        S = _s_to_numpy(result.S)
        for i in range(len(S) - 1):
            assert S[i] >= S[i + 1] - 1e-12


class TestSVDOrthogonality:
    """Tests for orthogonality of U and Vt."""

    def test_u_orthogonal(self):
        """U^T U should be identity."""
        np.random.seed(42)
        A = np.random.randn(5, 3)
        a = _make_matrix(A)
        result = svd(a)
        U = _to_numpy(result.U)
        UtU = U.T @ U
        assert np.allclose(UtU, np.eye(3), atol=1e-8)

    def test_vt_orthogonal(self):
        """Vt Vt^T should be identity."""
        np.random.seed(42)
        A = np.random.randn(5, 3)
        a = _make_matrix(A)
        result = svd(a)
        Vt = _to_numpy(result.Vt)
        VtVtT = Vt @ Vt.T
        assert np.allclose(VtVtT, np.eye(3), atol=1e-8)


class TestSVDShapes:
    """Tests for various matrix shapes."""

    def test_tall_matrix(self):
        """SVD of a tall matrix (m > n)."""
        np.random.seed(42)
        A = np.random.randn(6, 3)
        a = _make_matrix(A)
        result = svd(a)
        U = _to_numpy(result.U)
        S = _s_to_numpy(result.S)
        Vt = _to_numpy(result.Vt)
        assert U.shape == (6, 3)
        assert len(S) == 3
        assert Vt.shape == (3, 3)
        assert np.allclose(A, U @ np.diag(S) @ Vt, atol=1e-8)

    def test_fat_matrix(self):
        """SVD of a fat matrix (m < n)."""
        np.random.seed(42)
        A = np.random.randn(3, 6)
        a = _make_matrix(A)
        result = svd(a)
        U = _to_numpy(result.U)
        S = _s_to_numpy(result.S)
        Vt = _to_numpy(result.Vt)
        assert U.shape == (3, 3)
        assert len(S) == 3
        assert Vt.shape == (3, 6)
        assert np.allclose(A, U @ np.diag(S) @ Vt, atol=1e-8)

    def test_square_matrix(self):
        """SVD of a square matrix."""
        np.random.seed(42)
        A = np.random.randn(4, 4)
        a = _make_matrix(A)
        result = svd(a)
        U = _to_numpy(result.U)
        S = _s_to_numpy(result.S)
        Vt = _to_numpy(result.Vt)
        assert U.shape == (4, 4)
        assert Vt.shape == (4, 4)
        assert np.allclose(A, U @ np.diag(S) @ Vt, atol=1e-8)

    def test_1x1_matrix(self):
        """SVD of a 1x1 matrix."""
        A = np.array([[7.0]])
        a = _make_matrix(A)
        result = svd(a)
        S = _s_to_numpy(result.S)
        assert np.allclose(S, [7.0], atol=1e-10)

    def test_2x2_matrix(self):
        """SVD of a 2x2 matrix."""
        A = np.array([[3.0, 1.0], [1.0, 3.0]])
        a = _make_matrix(A)
        result = svd(a)
        U = _to_numpy(result.U)
        S = _s_to_numpy(result.S)
        Vt = _to_numpy(result.Vt)
        assert np.allclose(A, U @ np.diag(S) @ Vt, atol=1e-10)
        # Singular values of [[3,1],[1,3]] are 4 and 2
        assert np.allclose(S, [4.0, 2.0], atol=1e-10)


class TestSVDSpecialCases:
    """Tests for special and challenging matrices."""

    def test_rank_deficient(self):
        """SVD of a rank-deficient matrix."""
        A = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
        ])
        a = _make_matrix(A)
        result = svd(a)
        S = _s_to_numpy(result.S)
        # Rank 1 matrix: only one non-zero singular value
        assert S[0] > 1.0
        assert S[1] < 1e-8
        assert S[2] < 1e-8

    def test_zero_row(self):
        """SVD of a matrix with a zero row."""
        A = np.array([
            [1.0, 2.0],
            [0.0, 0.0],
            [3.0, 4.0],
        ])
        a = _make_matrix(A)
        result = svd(a)
        U = _to_numpy(result.U)
        S = _s_to_numpy(result.S)
        Vt = _to_numpy(result.Vt)
        assert np.allclose(A, U @ np.diag(S) @ Vt, atol=1e-8)

    def test_ill_conditioned(self):
        """SVD of an ill-conditioned matrix (large condition number)."""
        # Construct a matrix with specified singular values
        np.random.seed(42)
        U_ref, _ = np.linalg.qr(np.random.randn(4, 4))
        V_ref, _ = np.linalg.qr(np.random.randn(4, 4))
        S_ref = np.array([1e6, 1e3, 1.0, 1e-6])
        A = U_ref @ np.diag(S_ref) @ V_ref.T
        a = _make_matrix(A)
        result = svd(a)
        S = _s_to_numpy(result.S)
        assert np.allclose(np.sort(S)[::-1], np.sort(S_ref)[::-1], rtol=1e-4)

    def test_negative_entries(self):
        """SVD with all negative entries."""
        A = np.array([[-1.0, -2.0], [-3.0, -4.0]])
        a = _make_matrix(A)
        result = svd(a)
        U = _to_numpy(result.U)
        S = _s_to_numpy(result.S)
        Vt = _to_numpy(result.Vt)
        assert np.allclose(A, U @ np.diag(S) @ Vt, atol=1e-8)

    def test_comparison_with_numpy(self):
        """Singular values should match numpy's SVD."""
        np.random.seed(42)
        A = np.random.randn(5, 4)
        a = _make_matrix(A)
        result = svd(a)
        S_act = _s_to_numpy(result.S)
        _, S_np, _ = np.linalg.svd(A, full_matrices=False)
        assert np.allclose(S_act, S_np, rtol=1e-6)


class TestSVDInterface:
    """Tests for SVD interface and backward compatibility."""

    def test_returns_compensated_result(self):
        """Should return CompensatedSVDResult."""
        A = np.eye(2)
        a = _make_matrix(A)
        result = svd(a)
        assert isinstance(result, CompensatedSVDResult)

    def test_backward_compat_unpacking(self):
        """Tuple unpacking U, S, Vt = svd(a) should work."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        a = _make_matrix(A)
        U, S, Vt = svd(a)
        assert len(U) == 2
        assert len(S) == 2
        assert len(Vt) == 2

    def test_compensation_factors_populated(self):
        """compensation_factors should be a non-empty list."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        a = _make_matrix(A)
        result = svd(a)
        assert isinstance(result.compensation_factors, list)
        assert len(result.compensation_factors) > 0

    def test_empty_matrix_raises(self):
        """Empty matrix should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            svd([])

    def test_unknown_method_raises(self):
        """Unknown method should raise ValueError."""
        A = np.eye(2)
        a = _make_matrix(A)
        with pytest.raises(ValueError, match="Unknown SVD method"):
            svd(a, method="unknown")

    def test_reconstruction_error_attribute(self):
        """reconstruction_error should be a float."""
        A = np.eye(3)
        a = _make_matrix(A)
        result = svd(a)
        assert isinstance(result.reconstruction_error, float)
        assert result.reconstruction_error >= 0.0
