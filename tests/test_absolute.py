"""Tests for the AbsoluteValue class.

This module contains comprehensive tests for the AbsoluteValue class,
verifying mathematical operations, ACT compliance, and edge cases.
"""

import math
from decimal import Decimal
from typing import List, Tuple

import pytest

from balansis import ACT_ABSOLUTE_THRESHOLD, ACT_EPSILON
from balansis.core.absolute import AbsoluteValue


class TestAbsoluteValueCreation:
    """Test AbsoluteValue creation and validation."""

    def test_valid_creation(self):
        """Test creating valid AbsoluteValue instances."""
        # Standard values
        av1 = AbsoluteValue(magnitude=5.0, direction=1.0)
        assert av1.magnitude == 5.0
        assert av1.direction == 1.0

        # Zero magnitude (Absolute)
        av2 = AbsoluteValue(magnitude=0.0, direction=1)
        assert av2.magnitude == 0.0
        assert av2.direction == 1

        # Negative values
        av3 = AbsoluteValue(magnitude=3.0, direction=-1.0)
        assert av3.magnitude == 3.0
        assert av3.direction == -1.0

    def test_invalid_creation(self):
        """Test invalid AbsoluteValue creation raises errors."""
        # Negative magnitude
        with pytest.raises(ValueError):
            AbsoluteValue(magnitude=-1.0, direction=1.0)

        # Direction out of range
        with pytest.raises(ValueError):
            AbsoluteValue(magnitude=1.0, direction=2.0)

        with pytest.raises(ValueError):
            AbsoluteValue(magnitude=1.0, direction=-2.0)

    def test_from_float(self):
        """Test creating AbsoluteValue from float."""
        # Positive float
        av1 = AbsoluteValue.from_float(5.5)
        assert av1.magnitude == 5.5
        assert av1.direction == 1.0

        # Negative float
        av2 = AbsoluteValue.from_float(-3.2)
        assert av2.magnitude == 3.2
        assert av2.direction == -1.0

        # Zero becomes Absolute
        av3 = AbsoluteValue.from_float(0.0)
        assert av3.magnitude == 0.0
        assert av3.direction == 1

    def test_special_constructors(self):
        """Test special constructor methods."""
        # Absolute
        absolute = AbsoluteValue.absolute()
        assert absolute.magnitude == 0.0
        assert absolute.direction == 1

        # Unit positive
        unit_pos = AbsoluteValue.unit_positive()
        assert unit_pos.magnitude == 1.0
        assert unit_pos.direction == 1.0

        # Unit negative
        unit_neg = AbsoluteValue.unit_negative()
        assert unit_neg.magnitude == 1.0
        assert unit_neg.direction == -1.0


class TestAbsoluteValueProperties:
    """Test AbsoluteValue properties and checks."""

    def test_is_absolute(self):
        """Test is_absolute property."""
        absolute = AbsoluteValue(magnitude=0.0, direction=1)
        assert absolute.is_absolute()

        non_absolute = AbsoluteValue(magnitude=1.0, direction=1.0)
        assert not non_absolute.is_absolute()

    def test_is_positive(self):
        """Test is_positive property."""
        positive = AbsoluteValue(magnitude=5.0, direction=1.0)
        assert positive.is_positive()

        negative = AbsoluteValue(magnitude=5.0, direction=-1.0)
        assert not negative.is_positive()

        absolute = AbsoluteValue(magnitude=0.0, direction=1)
        assert not absolute.is_positive()

    def test_is_negative(self):
        """Test is_negative property."""
        negative = AbsoluteValue(magnitude=5.0, direction=-1.0)
        assert negative.is_negative()

        positive = AbsoluteValue(magnitude=5.0, direction=1.0)
        assert not positive.is_negative()

        absolute = AbsoluteValue(magnitude=0.0, direction=1)
        assert not absolute.is_negative()

    def test_is_compensating(self):
        """Test is_compensating property."""
        # Large magnitude values should be compensating
        large = AbsoluteValue(magnitude=1e10, direction=1.0)
        assert large.is_compensating()

        # Small magnitude values should not be compensating
        small = AbsoluteValue(magnitude=1.0, direction=1.0)
        assert not small.is_compensating()

        # Absolute is always compensating
        absolute = AbsoluteValue(magnitude=0.0, direction=1)
        assert absolute.is_compensating()


class TestAbsoluteValueArithmetic:
    """Test AbsoluteValue arithmetic operations."""

    def test_addition(self):
        """Test addition operations."""
        av1 = AbsoluteValue(magnitude=3.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=2.0, direction=1.0)

        result = av1 + av2
        assert result.magnitude == 5.0
        assert result.direction == 1.0

        # Addition with opposite directions
        av3 = AbsoluteValue(magnitude=2.0, direction=-1.0)
        result2 = av1 + av3
        assert result2.magnitude == 1.0
        assert result2.direction == 1.0

        # Addition resulting in Absolute
        av4 = AbsoluteValue(magnitude=3.0, direction=-1.0)
        result3 = av1 + av4
        assert result3.is_absolute()

    def test_subtraction(self):
        """Test subtraction operations."""
        av1 = AbsoluteValue(magnitude=5.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=2.0, direction=1.0)

        result = av1 - av2
        assert result.magnitude == 3.0
        assert result.direction == 1.0

        # Subtraction with negative direction
        av3 = AbsoluteValue(magnitude=2.0, direction=-1.0)
        result2 = av1 - av3
        assert result2.magnitude == 7.0
        assert result2.direction == 1.0

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        av = AbsoluteValue(magnitude=3.0, direction=1.0)

        # Positive scalar
        result1 = av * 2.0
        assert result1.magnitude == 6.0
        assert result1.direction == 1.0

        # Negative scalar
        result2 = av * -2.0
        assert result2.magnitude == 6.0
        assert result2.direction == -1.0

        # Zero scalar results in Absolute
        result3 = av * 0.0
        assert result3.is_absolute()

    def test_scalar_division(self):
        """Test scalar division."""
        av = AbsoluteValue(magnitude=6.0, direction=1.0)

        # Positive scalar
        result1 = av / 2.0
        assert result1.magnitude == 3.0
        assert result1.direction == 1.0

        # Negative scalar
        result2 = av / -2.0
        assert result2.magnitude == 3.0
        assert result2.direction == -1.0

        # Division by zero should raise error
        with pytest.raises(ValueError):
            av / 0.0

    def test_negation(self):
        """Test negation operation."""
        av = AbsoluteValue(magnitude=5.0, direction=1.0)
        neg_av = -av

        assert neg_av.magnitude == 5.0
        assert neg_av.direction == -1.0

        # Negating Absolute returns Absolute
        absolute = AbsoluteValue.absolute()
        neg_absolute = -absolute
        assert neg_absolute.is_absolute()

    def test_absolute_value(self):
        """Test absolute value operation."""
        positive = AbsoluteValue(magnitude=5.0, direction=1.0)
        negative = AbsoluteValue(magnitude=5.0, direction=-1.0)
        absolute = AbsoluteValue.absolute()

        assert abs(positive).magnitude == 5.0
        assert abs(positive).direction == 1.0

        assert abs(negative).magnitude == 5.0
        assert abs(negative).direction == 1.0

        assert abs(absolute).is_absolute()

    def test_inverse(self):
        """Test inverse operation."""
        av = AbsoluteValue(magnitude=4.0, direction=1.0)
        inv = av.inverse()

        assert abs(inv.magnitude - 0.25) < ACT_EPSILON
        assert inv.direction == 1.0

        # Inverse of Absolute should raise error
        absolute = AbsoluteValue.absolute()
        with pytest.raises(ValueError):
            absolute.inverse()


class TestAbsoluteValueComparison:
    """Test AbsoluteValue comparison operations."""

    def test_equality(self):
        """Test equality comparison."""
        av1 = AbsoluteValue(magnitude=5.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=5.0, direction=1.0)
        av3 = AbsoluteValue(magnitude=5.0, direction=-1.0)

        assert av1 == av2
        assert av1 != av3

        # Absolute equality
        abs1 = AbsoluteValue.absolute()
        abs2 = AbsoluteValue.absolute()
        assert abs1 == abs2

    def test_ordering(self):
        """Test ordering comparisons."""
        av1 = AbsoluteValue(magnitude=3.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=5.0, direction=1.0)
        av3 = AbsoluteValue(magnitude=3.0, direction=-1.0)

        assert av1 < av2
        assert av2 > av1
        assert av1 <= av2
        assert av2 >= av1

        # Negative values
        assert av3 < av1  # -3 < 3
        assert av1 > av3

    def test_hash(self):
        """Test hash functionality."""
        av1 = AbsoluteValue(magnitude=5.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=5.0, direction=1.0)
        av3 = AbsoluteValue(magnitude=5.0, direction=-1.0)

        # Equal objects should have equal hashes
        assert hash(av1) == hash(av2)

        # Different objects should have different hashes
        assert hash(av1) != hash(av3)

        # Should be usable in sets and dicts
        av_set = {av1, av2, av3}
        assert len(av_set) == 2  # av1 and av2 are equal


class TestAbsoluteValueConversion:
    """Test AbsoluteValue conversion operations."""

    def test_to_float(self):
        """Test conversion to float."""
        positive = AbsoluteValue(magnitude=5.0, direction=1.0)
        assert positive.to_float() == 5.0

        negative = AbsoluteValue(magnitude=5.0, direction=-1.0)
        assert negative.to_float() == -5.0

        absolute = AbsoluteValue.absolute()
        assert absolute.to_float() == 0.0

    def test_string_representation(self):
        """Test string representations."""
        positive = AbsoluteValue(magnitude=5.0, direction=1.0)
        negative = AbsoluteValue(magnitude=5.0, direction=-1.0)
        absolute = AbsoluteValue.absolute()

        assert str(positive) == "AbsoluteValue(5.0, +)"
        assert str(negative) == "AbsoluteValue(5.0, -)"
        assert str(absolute) == "AbsoluteValue(Absolute)"

        assert repr(positive) == "AbsoluteValue(magnitude=5.0, direction=1.0)"
        assert repr(absolute) == "AbsoluteValue(magnitude=0.0, direction=1)"


class TestAbsoluteValueEdgeCases:
    """Test AbsoluteValue edge cases and error conditions."""

    def test_very_large_values(self):
        """Test handling of very large values."""
        large = AbsoluteValue(magnitude=1e100, direction=1.0)
        assert large.is_compensating()

        # Operations with large values should not overflow
        result = large + large
        assert result.magnitude == 2e100
        assert result.direction == 1.0

    def test_very_small_values(self):
        """Test handling of very small values."""
        small = AbsoluteValue(magnitude=1e-100, direction=1.0)
        assert not small.is_absolute()

        # Operations with small values should maintain precision
        result = small + small
        assert result.magnitude == 2e-100
        assert result.direction == 1.0

    def test_precision_limits(self):
        """Test behavior at precision limits."""
        # Values close to ACT_EPSILON
        tiny = AbsoluteValue(magnitude=ACT_EPSILON / 2, direction=1.0)
        result = tiny + tiny
        assert result.magnitude == ACT_EPSILON

    def test_mathematical_consistency(self):
        """Test mathematical consistency properties."""
        av = AbsoluteValue(magnitude=5.0, direction=1.0)

        # Additive identity
        absolute = AbsoluteValue.absolute()
        assert av + absolute == av
        assert absolute + av == av

        # Additive inverse
        neg_av = -av
        result = av + neg_av
        assert result.is_absolute()

        # Multiplicative identity
        unit = AbsoluteValue(magnitude=1.0, direction=1.0)
        assert (av * 1.0) == av

        # Multiplicative inverse
        inv = av.inverse()
        product = av * inv.to_float()
        assert abs(product.magnitude - 1.0) < ACT_EPSILON


class TestAbsoluteValueACTCompliance:
    """Test ACT (Absolute Compensation Theory) compliance."""

    def test_compensation_axiom(self):
        """Test that operations maintain compensation."""
        # Large values should trigger compensation
        large1 = AbsoluteValue(magnitude=1e10, direction=1.0)
        large2 = AbsoluteValue(magnitude=1e10, direction=-1.0)

        # Addition should result in compensation
        result = large1 + large2
        assert result.is_absolute()  # Perfect compensation

    def test_stability_axiom(self):
        """Test mathematical stability."""
        # Operations should not produce unstable results
        av = AbsoluteValue(magnitude=1e-10, direction=1.0)

        # Repeated operations should remain stable
        result = av
        for _ in range(1000):
            result = result + av

        expected_magnitude = 1001 * 1e-10
        assert abs(result.magnitude - expected_magnitude) < ACT_EPSILON

    def test_eternity_axiom(self):
        """Test that ratios remain invariant."""
        av1 = AbsoluteValue(magnitude=6.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=3.0, direction=1.0)

        # Ratio should be preserved under scaling
        scale = 2.0
        scaled_av1 = av1 * scale
        scaled_av2 = av2 * scale

        original_ratio = av1.magnitude / av2.magnitude
        scaled_ratio = scaled_av1.magnitude / scaled_av2.magnitude

        assert abs(original_ratio - scaled_ratio) < ACT_EPSILON


if __name__ == "__main__":
    pytest.main([__file__])
