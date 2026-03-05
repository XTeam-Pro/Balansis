"""Tests for the GEMM (general matrix multiply) module and linalg exports."""

import pytest

from balansis.core.absolute import AbsoluteValue
from balansis.linalg.gemm import matmul
from balansis import ACT_EPSILON


class TestMatmul:
    """Test matrix multiplication."""

    def test_matmul_basic_2x2(self):
        """Test basic 2x2 matrix multiplication."""
        a = [
            [AbsoluteValue(magnitude=1.0, direction=1), AbsoluteValue(magnitude=2.0, direction=1)],
            [AbsoluteValue(magnitude=3.0, direction=1), AbsoluteValue(magnitude=4.0, direction=1)],
        ]
        b = [
            [AbsoluteValue(magnitude=5.0, direction=1), AbsoluteValue(magnitude=6.0, direction=1)],
            [AbsoluteValue(magnitude=7.0, direction=1), AbsoluteValue(magnitude=8.0, direction=1)],
        ]
        result, compensation = matmul(a, b)
        assert len(result) == 2
        assert len(result[0]) == 2
        assert isinstance(compensation, float)
        # C[0][0] = 1*5 + 2*7 = 19
        assert abs(result[0][0].to_float() - 19.0) < ACT_EPSILON

    def test_matmul_returns_tuple(self):
        """Test that matmul returns (matrix, compensation) tuple."""
        a = [[AbsoluteValue(magnitude=2.0, direction=1)]]
        b = [[AbsoluteValue(magnitude=3.0, direction=1)]]
        result = matmul(a, b)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_matmul_1x1(self):
        """Test 1x1 matrix multiplication."""
        a = [[AbsoluteValue(magnitude=4.0, direction=1)]]
        b = [[AbsoluteValue(magnitude=5.0, direction=1)]]
        result, comp = matmul(a, b)
        assert abs(result[0][0].to_float() - 20.0) < ACT_EPSILON

    def test_matmul_empty_matrices(self):
        """Test matmul with empty matrices."""
        result, comp = matmul([], [])
        assert result == []
        assert comp == 1.0

    def test_matmul_invalid_shape_a(self):
        """Test matmul with inconsistent rows in matrix a."""
        a = [
            [AbsoluteValue(magnitude=1.0, direction=1)],
            [AbsoluteValue(magnitude=1.0, direction=1), AbsoluteValue(magnitude=2.0, direction=1)],
        ]
        b = [[AbsoluteValue(magnitude=1.0, direction=1)]]
        with pytest.raises(ValueError, match="Invalid shape for matrix a"):
            matmul(a, b)

    def test_matmul_inner_dimension_mismatch(self):
        """Test matmul with incompatible inner dimensions."""
        a = [[AbsoluteValue(magnitude=1.0, direction=1)] * 3]
        b = [[AbsoluteValue(magnitude=1.0, direction=1)] * 2]
        with pytest.raises(ValueError, match="Inner dimensions must match"):
            matmul(a, b)

    def test_matmul_invalid_shape_b(self):
        """Test matmul with inconsistent rows in matrix b."""
        a = [[AbsoluteValue(magnitude=1.0, direction=1)]]
        b = [
            [AbsoluteValue(magnitude=1.0, direction=1)],
            [AbsoluteValue(magnitude=1.0, direction=1), AbsoluteValue(magnitude=2.0, direction=1)],
        ]
        with pytest.raises(ValueError):
            matmul(a, b)

    def test_matmul_without_compensation(self):
        """Test matmul with compensation disabled."""
        a = [[AbsoluteValue(magnitude=2.0, direction=1)]]
        b = [[AbsoluteValue(magnitude=3.0, direction=1)]]
        result, comp = matmul(a, b, use_compensation=False)
        assert abs(result[0][0].to_float() - 6.0) < ACT_EPSILON

    def test_matmul_identity_matrix(self):
        """Test matmul with identity matrix preserves values."""
        identity = [
            [AbsoluteValue(magnitude=1.0, direction=1), AbsoluteValue.absolute()],
            [AbsoluteValue.absolute(), AbsoluteValue(magnitude=1.0, direction=1)],
        ]
        a = [
            [AbsoluteValue(magnitude=3.0, direction=1), AbsoluteValue(magnitude=4.0, direction=1)],
            [AbsoluteValue(magnitude=5.0, direction=1), AbsoluteValue(magnitude=6.0, direction=1)],
        ]
        result, comp = matmul(a, identity)
        assert abs(result[0][0].to_float() - 3.0) < ACT_EPSILON
        assert abs(result[1][1].to_float() - 6.0) < ACT_EPSILON

    def test_matmul_with_negative_directions(self):
        """Test matmul with mixed direction values."""
        a = [[AbsoluteValue(magnitude=3.0, direction=-1)]]
        b = [[AbsoluteValue(magnitude=4.0, direction=-1)]]
        result, comp = matmul(a, b)
        # (-3) * (-4) = 12
        assert abs(result[0][0].to_float() - 12.0) < ACT_EPSILON


class TestLinalgExports:
    """Test linalg module exports."""

    def test_import_matmul(self):
        """Test matmul is importable from linalg."""
        from balansis.linalg import matmul as mm
        assert callable(mm)

    def test_import_svd(self):
        """Test svd is importable from linalg."""
        from balansis.linalg import svd
        assert callable(svd)

    def test_import_qr_decompose(self):
        """Test qr_decompose is importable from linalg."""
        from balansis.linalg import qr_decompose
        assert callable(qr_decompose)
