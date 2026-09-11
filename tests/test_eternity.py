"""Tests for the EternalRatio class.

This module contains comprehensive tests for the EternalRatio class,
verifying structural ratio calculations, stability properties, and ACT compliance.
"""

import math
from decimal import Decimal
from typing import List, Tuple

import pytest

from balansis import ACT_EPSILON, ACT_STABILITY_THRESHOLD
from balansis.core.absolute import AbsoluteValue
from balansis.core.eternity import EternalRatio


class TestEternalRatioCreation:
    """Test EternalRatio creation and validation."""

    def test_valid_creation(self):
        """Test creating valid EternalRatio instances."""
        numerator = AbsoluteValue(magnitude=6.0, direction=1.0)
        denominator = AbsoluteValue(magnitude=3.0, direction=1.0)

        ratio = EternalRatio(numerator=numerator, denominator=denominator)
        assert ratio.numerator == numerator
        assert ratio.denominator == denominator

    def test_absolute_denominator_error(self):
        """Test that Absolute denominator raises error."""
        numerator = AbsoluteValue(magnitude=6.0, direction=1.0)
        absolute_denom = AbsoluteValue.absolute()

        with pytest.raises(ValueError, match="Denominator cannot be Absolute"):
            EternalRatio(numerator=numerator, denominator=absolute_denom)

    def test_from_values(self):
        """Test creating EternalRatio from float values."""
        ratio = EternalRatio.from_values(6.0, 3.0)

        assert ratio.numerator.magnitude == 6.0
        assert ratio.numerator.direction == 1.0
        assert ratio.denominator.magnitude == 3.0
        assert ratio.denominator.direction == 1.0

    def test_from_values_with_negatives(self):
        """Test creating EternalRatio from negative values."""
        ratio = EternalRatio.from_values(-6.0, 3.0)

        assert ratio.numerator.magnitude == 6.0
        assert ratio.numerator.direction == -1.0
        assert ratio.denominator.magnitude == 3.0
        assert ratio.denominator.direction == 1.0

    def test_from_values_zero_denominator(self):
        """Test that zero denominator raises error."""
        with pytest.raises(ValueError):
            EternalRatio.from_values(6.0, 0.0)

    def test_unity(self):
        """Test unity ratio creation."""
        unity = EternalRatio.unity()

        assert unity.numerator.magnitude == 1.0
        assert unity.numerator.direction == 1.0
        assert unity.denominator.magnitude == 1.0
        assert unity.denominator.direction == 1.0
        assert unity.numerical_value() == 1.0


class TestEternalRatioProperties:
    """Test EternalRatio properties and calculations."""

    def test_numerical_value(self):
        """Test numerical value calculation."""
        # Positive ratio
        ratio1 = EternalRatio.from_values(6.0, 3.0)
        assert ratio1.numerical_value() == 2.0

        # Negative numerator
        ratio2 = EternalRatio.from_values(-6.0, 3.0)
        assert ratio2.numerical_value() == -2.0

        # Negative denominator
        ratio3 = EternalRatio.from_values(6.0, -3.0)
        assert ratio3.numerical_value() == -2.0

        # Both negative
        ratio4 = EternalRatio.from_values(-6.0, -3.0)
        assert ratio4.numerical_value() == 2.0

    def test_signed_value(self):
        """Test signed value calculation."""
        # Positive ratio
        ratio1 = EternalRatio.from_values(6.0, 3.0)
        assert ratio1.signed_value() == 1.0

        # Negative ratio
        ratio2 = EternalRatio.from_values(-6.0, 3.0)
        assert ratio2.signed_value() == -1.0

        # Unity ratio
        unity = EternalRatio.unity()
        assert unity.signed_value() == 1.0

    def test_is_stable(self):
        """Test stability check."""
        # Stable ratio (close to 1)
        stable = EternalRatio.from_values(1.1, 1.0)
        assert stable.is_stable()

        # Unstable ratio (far from 1)
        unstable = EternalRatio.from_values(10.0, 1.0)
        assert not unstable.is_stable()

        # Unity is always stable
        unity = EternalRatio.unity()
        assert unity.is_stable()

    def test_is_unity(self):
        """Test unity check."""
        unity = EternalRatio.unity()
        assert unity.is_unity()

        non_unity = EternalRatio.from_values(2.0, 1.0)
        assert not non_unity.is_unity()

        # Close to unity should be considered unity
        close_unity = EternalRatio.from_values(1.0 + ACT_EPSILON / 2, 1.0)
        assert close_unity.is_unity()

    def test_is_reciprocal(self):
        """Test reciprocal relationship check."""
        ratio1 = EternalRatio.from_values(3.0, 2.0)
        ratio2 = EternalRatio.from_values(2.0, 3.0)

        assert ratio1.is_reciprocal(ratio2)
        assert ratio2.is_reciprocal(ratio1)

        # Non-reciprocal ratios
        ratio3 = EternalRatio.from_values(4.0, 2.0)
        assert not ratio1.is_reciprocal(ratio3)


class TestEternalRatioComparison:
    """Test EternalRatio comparison operations."""

    def test_equality(self):
        """Test equality comparison."""
        ratio1 = EternalRatio.from_values(6.0, 3.0)
        ratio2 = EternalRatio.from_values(6.0, 3.0)
        ratio3 = EternalRatio.from_values(4.0, 2.0)  # Same numerical value
        ratio4 = EternalRatio.from_values(8.0, 3.0)  # Different value

        assert ratio1 == ratio2
        assert ratio1 == ratio3  # Same numerical value
        assert ratio1 != ratio4

    def test_ordering(self):
        """Test ordering comparisons."""
        ratio1 = EternalRatio.from_values(2.0, 1.0)  # 2.0
        ratio2 = EternalRatio.from_values(6.0, 2.0)  # 3.0
        ratio3 = EternalRatio.from_values(-4.0, 2.0)  # -2.0

        assert ratio1 < ratio2
        assert ratio2 > ratio1
        assert ratio3 < ratio1
        assert ratio1 > ratio3

        assert ratio1 <= ratio2
        assert ratio2 >= ratio1

    def test_hash(self):
        """Test hash functionality."""
        ratio1 = EternalRatio.from_values(6.0, 3.0)
        ratio2 = EternalRatio.from_values(6.0, 3.0)
        ratio3 = EternalRatio.from_values(4.0, 2.0)  # Same numerical value

        # Equal ratios should have equal hashes
        assert hash(ratio1) == hash(ratio2)
        assert hash(ratio1) == hash(ratio3)

        # Should be usable in sets and dicts
        ratio_set = {ratio1, ratio2, ratio3}
        assert len(ratio_set) == 1  # All are equal


class TestEternalRatioArithmetic:
    """Test EternalRatio arithmetic operations."""

    def test_multiplication(self):
        """Test ratio multiplication."""
        ratio1 = EternalRatio.from_values(3.0, 2.0)  # 1.5
        ratio2 = EternalRatio.from_values(4.0, 3.0)  # 1.333...

        result = ratio1 * ratio2
        expected = 1.5 * (4.0 / 3.0)  # 2.0

        assert abs(result.numerical_value() - expected) < ACT_EPSILON

    def test_division(self):
        """Test ratio division."""
        ratio1 = EternalRatio.from_values(6.0, 2.0)  # 3.0
        ratio2 = EternalRatio.from_values(3.0, 2.0)  # 1.5

        result = ratio1 / ratio2
        expected = 3.0 / 1.5  # 2.0

        assert abs(result.numerical_value() - expected) < ACT_EPSILON

    def test_addition(self):
        """Test ratio addition."""
        ratio1 = EternalRatio.from_values(3.0, 2.0)  # 1.5
        ratio2 = EternalRatio.from_values(1.0, 2.0)  # 0.5

        result = ratio1 + ratio2
        expected = 1.5 + 0.5  # 2.0

        assert abs(result.numerical_value() - expected) < ACT_EPSILON

    def test_subtraction(self):
        """Test ratio subtraction."""
        ratio1 = EternalRatio.from_values(5.0, 2.0)  # 2.5
        ratio2 = EternalRatio.from_values(1.0, 2.0)  # 0.5

        result = ratio1 - ratio2
        expected = 2.5 - 0.5  # 2.0

        assert abs(result.numerical_value() - expected) < ACT_EPSILON

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        ratio = EternalRatio.from_values(3.0, 2.0)  # 1.5
        scalar = 2.0

        result = ratio * scalar
        expected = 1.5 * 2.0  # 3.0

        assert abs(result.numerical_value() - expected) < ACT_EPSILON

    def test_scalar_division(self):
        """Test scalar division."""
        ratio = EternalRatio.from_values(6.0, 2.0)  # 3.0
        scalar = 2.0

        result = ratio / scalar
        expected = 3.0 / 2.0  # 1.5

        assert abs(result.numerical_value() - expected) < ACT_EPSILON

    def test_power(self):
        """Test power operation."""
        ratio = EternalRatio.from_values(4.0, 2.0)  # 2.0

        # Positive power
        result1 = ratio**3
        expected1 = 2.0**3  # 8.0
        assert abs(result1.numerical_value() - expected1) < ACT_EPSILON

        # Negative power (reciprocal)
        result2 = ratio**-1
        expected2 = 1.0 / 2.0  # 0.5
        assert abs(result2.numerical_value() - expected2) < ACT_EPSILON

        # Zero power should give unity
        result3 = ratio**0
        assert result3.is_unity()

    def test_inverse(self):
        """Test inverse operation."""
        ratio = EternalRatio.from_values(6.0, 3.0)  # 2.0
        inverse = ratio.inverse()

        expected = 1.0 / 2.0  # 0.5
        assert abs(inverse.numerical_value() - expected) < ACT_EPSILON

        # Inverse of inverse should be original
        double_inverse = inverse.inverse()
        assert (
            abs(double_inverse.numerical_value() - ratio.numerical_value())
            < ACT_EPSILON
        )

    def test_simplify(self):
        """Test ratio simplification."""
        # Create ratio with common factors
        numerator = AbsoluteValue(magnitude=12.0, direction=1.0)
        denominator = AbsoluteValue(magnitude=8.0, direction=1.0)
        ratio = EternalRatio(numerator=numerator, denominator=denominator)

        simplified = ratio.simplify()

        # Should maintain the same numerical value
        assert abs(simplified.numerical_value() - ratio.numerical_value()) < ACT_EPSILON

        # Simplified form should have smaller magnitudes
        assert simplified.numerator.magnitude <= ratio.numerator.magnitude
        assert simplified.denominator.magnitude <= ratio.denominator.magnitude


class TestEternalRatioStringRepresentation:
    """Test EternalRatio string representations."""

    def test_str_representation(self):
        """Test string representation."""
        ratio = EternalRatio.from_values(6.0, 3.0)
        str_repr = str(ratio)

        assert "6.0" in str_repr
        assert "3.0" in str_repr
        assert "EternalRatio" in str_repr

    def test_repr_representation(self):
        """Test repr representation."""
        ratio = EternalRatio.from_values(6.0, 3.0)
        repr_str = repr(ratio)

        assert "EternalRatio" in repr_str
        assert "numerator" in repr_str
        assert "denominator" in repr_str


class TestEternalRatioEdgeCases:
    """Test EternalRatio edge cases and error conditions."""

    def test_very_large_ratios(self):
        """Test handling of very large ratios."""
        large_ratio = EternalRatio.from_values(1e100, 1.0)
        assert not large_ratio.is_stable()

        # Operations should not overflow
        result = large_ratio * 2.0
        assert result.numerical_value() == 2e100

    def test_very_small_ratios(self):
        """Test handling of very small ratios."""
        small_ratio = EternalRatio.from_values(1e-100, 1.0)
        assert not small_ratio.is_stable()

        # Operations should maintain precision
        result = small_ratio * 2.0
        assert result.numerical_value() == 2e-100

    def test_precision_limits(self):
        """Test behavior at precision limits."""
        # Ratios close to unity
        close_unity = EternalRatio.from_values(1.0 + ACT_EPSILON / 2, 1.0)
        assert close_unity.is_unity()

        # Ratios at stability threshold
        threshold_ratio = EternalRatio.from_values(1.0 + ACT_STABILITY_THRESHOLD, 1.0)
        assert not threshold_ratio.is_stable()

    def test_absolute_numerator_handling(self):
        """Test handling of Absolute numerator."""
        absolute_num = AbsoluteValue.absolute()
        denominator = AbsoluteValue(magnitude=3.0, direction=1.0)

        ratio = EternalRatio(numerator=absolute_num, denominator=denominator)
        assert ratio.numerical_value() == 0.0
        assert ratio.signed_value() == 0.0


class TestEternalRatioMathematicalProperties:
    """Test mathematical properties of EternalRatio."""

    def test_multiplicative_identity(self):
        """Test multiplicative identity property."""
        ratio = EternalRatio.from_values(5.0, 3.0)
        unity = EternalRatio.unity()

        result1 = ratio * unity
        result2 = unity * ratio

        assert abs(result1.numerical_value() - ratio.numerical_value()) < ACT_EPSILON
        assert abs(result2.numerical_value() - ratio.numerical_value()) < ACT_EPSILON

    def test_multiplicative_inverse(self):
        """Test multiplicative inverse property."""
        ratio = EternalRatio.from_values(5.0, 3.0)
        inverse = ratio.inverse()

        result = ratio * inverse
        assert result.is_unity()

    def test_associativity(self):
        """Test associativity of multiplication."""
        ratio1 = EternalRatio.from_values(2.0, 1.0)
        ratio2 = EternalRatio.from_values(3.0, 2.0)
        ratio3 = EternalRatio.from_values(5.0, 3.0)

        result1 = (ratio1 * ratio2) * ratio3
        result2 = ratio1 * (ratio2 * ratio3)

        assert abs(result1.numerical_value() - result2.numerical_value()) < ACT_EPSILON

    def test_commutativity(self):
        """Test commutativity of multiplication."""
        ratio1 = EternalRatio.from_values(3.0, 2.0)
        ratio2 = EternalRatio.from_values(5.0, 4.0)

        result1 = ratio1 * ratio2
        result2 = ratio2 * ratio1

        assert abs(result1.numerical_value() - result2.numerical_value()) < ACT_EPSILON

    def test_distributivity(self):
        """Test distributivity over addition."""
        ratio1 = EternalRatio.from_values(2.0, 1.0)
        ratio2 = EternalRatio.from_values(3.0, 2.0)
        ratio3 = EternalRatio.from_values(5.0, 4.0)

        # a * (b + c) = (a * b) + (a * c)
        left = ratio1 * (ratio2 + ratio3)
        right = (ratio1 * ratio2) + (ratio1 * ratio3)

        assert abs(left.numerical_value() - right.numerical_value()) < ACT_EPSILON


class TestEternalRatioACTCompliance:
    """Test ACT (Absolute Compensation Theory) compliance."""

    def test_structural_invariance(self):
        """Test that structural ratios remain invariant."""
        ratio = EternalRatio.from_values(6.0, 3.0)
        original_value = ratio.numerical_value()

        # Scale both numerator and denominator
        scale = 2.0
        scaled_num = ratio.numerator * scale
        scaled_denom = ratio.denominator * scale
        scaled_ratio = EternalRatio(numerator=scaled_num, denominator=scaled_denom)

        # Ratio should remain the same
        assert abs(scaled_ratio.numerical_value() - original_value) < ACT_EPSILON

    def test_compensation_stability(self):
        """Test stability under compensation."""
        # Create unstable ratio
        unstable = EternalRatio.from_values(1e10, 1.0)
        assert not unstable.is_stable()

        # Apply compensation through operations
        compensated = unstable * EternalRatio.from_values(1.0, 1e10)
        assert compensated.is_unity()
        assert compensated.is_stable()

    def test_eternity_preservation(self):
        """Test that eternal ratios preserve their essence."""
        ratio = EternalRatio.from_values(math.pi, math.e)

        # Multiple operations should preserve the fundamental ratio
        transformed = ratio.inverse().inverse()
        assert (
            abs(transformed.numerical_value() - ratio.numerical_value()) < ACT_EPSILON
        )

        # Power operations should maintain structural relationships
        squared = ratio**2
        sqrt_squared = squared**0.5
        assert (
            abs(sqrt_squared.numerical_value() - ratio.numerical_value())
            < ACT_EPSILON * 10
        )  # Allow for floating point errors


if __name__ == "__main__":
    pytest.main([__file__])
