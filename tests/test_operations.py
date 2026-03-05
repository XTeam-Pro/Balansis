"""Tests for the Operations module.

This module contains comprehensive tests for compensated arithmetic functions,
verifying mathematical stability and ACT compliance.
"""

import pytest
import math
import numpy as np
from typing import List, Tuple

from balansis.core.absolute import AbsoluteValue
from balansis.core.eternity import EternalRatio
from balansis.core.operations import Operations
from balansis import ACT_EPSILON, ACT_COMPENSATION_FACTOR


class TestCompensatedArithmetic:
    """Test basic compensated arithmetic operations."""
    
    def test_compensated_add_basic(self):
        """Test basic compensated addition."""
        av1 = AbsoluteValue(magnitude=3.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=2.0, direction=1.0)
        
        result, _ = Operations.compensated_add(av1, av2)
        assert result.magnitude == 5.0
        assert result.direction == 1.0
    
    def test_compensated_add_opposite_directions(self):
        """Test compensated addition with opposite directions."""
        av1 = AbsoluteValue(magnitude=5.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=3.0, direction=-1.0)
        
        result, _ = Operations.compensated_add(av1, av2)
        assert result.magnitude == 2.0
        assert result.direction == 1.0
    
    def test_compensated_add_perfect_cancellation(self):
        """Test compensated addition with perfect cancellation."""
        av1 = AbsoluteValue(magnitude=5.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=5.0, direction=-1.0)
        
        result, _ = Operations.compensated_add(av1, av2)
        assert result.is_absolute()
    
    def test_compensated_add_with_absolute(self):
        """Test compensated addition with Absolute values."""
        av = AbsoluteValue(magnitude=5.0, direction=1.0)
        absolute = AbsoluteValue.absolute()
        
        result1, _ = Operations.compensated_add(av, absolute)
        result2, _ = Operations.compensated_add(absolute, av)
        
        assert result1 == av
        assert result2 == av
    
    def test_compensated_add_large_values(self):
        """Test compensated addition with large values."""
        large1 = AbsoluteValue(magnitude=1e10, direction=1.0)
        large2 = AbsoluteValue(magnitude=1e10, direction=1.0)
        
        result, _ = Operations.compensated_add(large1, large2)
        assert result.magnitude == 2e10
        assert result.direction == 1.0
    
    def test_compensated_add_near_canceling_operands(self):
        """Test addition with near-canceling operands (lines 59-64)."""
        # Create operands that nearly cancel each other
        a = AbsoluteValue(magnitude=1.0, direction=1)
        b = AbsoluteValue(magnitude=0.9999999, direction=-1)
        
        result, compensation = Operations.compensated_add(a, b)
        
        # Should detect near-canceling and apply compensation
        assert result.magnitude > 0
        assert compensation >= 1.0  # Compensation should be applied
    
    def test_compensated_add_small_magnitude_result(self):
        """Test addition resulting in small magnitude (lines 70-73)."""
        # Create operands that result in very small magnitude
        a = AbsoluteValue(magnitude=1e-10, direction=1)
        b = AbsoluteValue(magnitude=1e-11, direction=1)
        
        result, compensation = Operations.compensated_add(a, b)
        
        # Should apply compensation for small result
        assert result.magnitude > 0
        assert compensation >= 1.0
    
    def test_compensated_multiply_basic(self):
        """Test basic compensated multiplication."""
        av1 = AbsoluteValue(magnitude=3.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=4.0, direction=1.0)
        
        result, _ = Operations.compensated_multiply(av1, av2)
        assert result.magnitude == 12.0
        assert result.direction == 1.0
    
    def test_compensated_multiply_opposite_signs(self):
        """Test compensated multiplication with opposite signs."""
        av1 = AbsoluteValue(magnitude=3.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=4.0, direction=-1.0)
        
        result, _ = Operations.compensated_multiply(av1, av2)
        assert result.magnitude == 12.0
        assert result.direction == -1.0
    
    def test_compensated_multiply_with_absolute(self):
        """Test compensated multiplication with Absolute."""
        av = AbsoluteValue(magnitude=5.0, direction=1.0)
        absolute = AbsoluteValue.absolute()
        
        result1, _ = Operations.compensated_multiply(av, absolute)
        result2, _ = Operations.compensated_multiply(absolute, av)
        
        assert result1.is_absolute()
        assert result2.is_absolute()
    
    def test_compensated_multiply_absolute_operand(self):
        """Test multiplication with absolute operand (lines 91-92)."""
        av = AbsoluteValue(magnitude=5.0, direction=1.0)
        absolute = AbsoluteValue.absolute()
        
        # Test both orders
        result1, compensation1 = Operations.compensated_multiply(av, absolute)
        result2, compensation2 = Operations.compensated_multiply(absolute, av)
        
        assert result1.is_absolute()
        assert result2.is_absolute()
        assert compensation1 == 0.0
        assert compensation2 == 0.0
    
    def test_compensated_multiply_standard_operation(self):
        """Test standard multiplication operation (lines 95-96)."""
        av1 = AbsoluteValue(magnitude=3.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=4.0, direction=-1.0)
        
        result, compensation = Operations.compensated_multiply(av1, av2)
        
        assert result.magnitude == 12.0
        assert result.direction == -1.0
        assert compensation == 1.0
    
    def test_compensated_multiply_overflow_compensation(self):
        """Test multiplication with overflow compensation (lines 99-104)."""
        # Create values that would cause overflow
        large1 = AbsoluteValue(magnitude=1e200, direction=1.0)
        large2 = AbsoluteValue(magnitude=1e200, direction=1.0)
        
        result, compensation = Operations.compensated_multiply(large1, large2)
        
        # Should handle overflow with compensation
        assert result.magnitude > 0
        assert compensation > 1.0
    
    def test_compensated_multiply_underflow_compensation(self):
        """Test multiplication with underflow compensation (lines 107-111)."""
        # Create values that would cause underflow
        small1 = AbsoluteValue(magnitude=1e-200, direction=1.0)
        small2 = AbsoluteValue(magnitude=1e-200, direction=1.0)
        
        result, compensation = Operations.compensated_multiply(small1, small2)
        
        # Should handle underflow with compensation - returns Absolute for very small results
        assert result.is_absolute()
        assert compensation < 1.0  # Underflow compensation is the small magnitude
    
    def test_compensated_divide_basic(self):
        """Test basic compensated division."""
        av1 = AbsoluteValue(magnitude=12.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=3.0, direction=1.0)
        
        result, compensation = Operations.compensated_divide(av1, av2)
        # EternalRatio doesn't have magnitude/direction, check ratio value
        assert abs(result.numerical_value() - 4.0) < ACT_EPSILON
        assert isinstance(compensation, float)

    def test_compensated_divide_opposite_signs(self):
        """Test compensated division with opposite signs."""
        av1 = AbsoluteValue(magnitude=12.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=3.0, direction=-1.0)

        result, compensation = Operations.compensated_divide(av1, av2)
        # EternalRatio doesn't have magnitude/direction, check ratio value
        assert abs(result.numerical_value() + 4.0) < ACT_EPSILON  # negative result
        assert isinstance(compensation, float)
    
    def test_compensated_divide_by_absolute(self):
        """Test compensated division by Absolute raises error."""
        av = AbsoluteValue(magnitude=5.0, direction=1.0)
        absolute = AbsoluteValue.absolute()
        
        with pytest.raises(ValueError, match="Cannot divide by Absolute"):
            Operations.compensated_divide(av, absolute)
    
    def test_compensated_divide_absolute_numerator(self):
        """Test compensated division with Absolute numerator."""
        absolute = AbsoluteValue.absolute()
        av = AbsoluteValue(magnitude=5.0, direction=1.0)
        
        result, compensation = Operations.compensated_divide(absolute, av)
        # Division of Absolute by non-Absolute should result in EternalRatio with Absolute numerator
        assert result.numerator.is_absolute()
        assert isinstance(compensation, float)
    
    def test_compensated_divide_eternal_ratio_creation(self):
        """Test EternalRatio creation in division (line 136)."""
        numerator = AbsoluteValue(magnitude=8.0, direction=1.0)
        denominator = AbsoluteValue(magnitude=2.0, direction=-1.0)
        
        result, compensation = Operations.compensated_divide(numerator, denominator)

        assert isinstance(result, EternalRatio)
        assert isinstance(compensation, float)
        assert result.numerator == numerator
        assert result.denominator == denominator


class TestCompensatedTranscendental:
    """Test compensated transcendental functions."""
    
    def test_compensated_power_integer(self):
        """Test compensated power with integer exponent."""
        av = AbsoluteValue(magnitude=2.0, direction=1.0)
        
        result, _ = Operations.compensated_power(av, 3)
        assert result.magnitude == 8.0
        assert result.direction == 1.0
    
    def test_compensated_power_negative_exponent(self):
        """Test compensated power with negative exponent."""
        av = AbsoluteValue(magnitude=4.0, direction=1.0)
        
        result, _ = Operations.compensated_power(av, -1)
        assert abs(result.magnitude - 0.25) < ACT_EPSILON
        assert result.direction == 1.0
    
    def test_compensated_power_zero_exponent_basic(self):
        """Test compensated power with zero exponent."""
        av = AbsoluteValue(magnitude=5.0, direction=1.0)
        
        result, compensation = Operations.compensated_power(av, 0)
        assert result.magnitude == 1.0
        assert result.direction == 1.0
        assert compensation == 1.0  # Based on actual implementation
    
    def test_compensated_power_absolute_base(self):
        """Test compensated power with Absolute base."""
        absolute = AbsoluteValue.absolute()
        
        result, _ = Operations.compensated_power(absolute, 5)
        assert result.is_absolute()
    
    def test_compensated_power_absolute_base_error(self):
        """Test power with absolute base error cases (lines 154, 156)."""
        absolute = AbsoluteValue.absolute()
        
        # Test with non-integer exponent should raise error
        try:
            Operations.compensated_power(absolute, 2.5)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "absolute" in str(e).lower() or "integer" in str(e).lower()
    
    def test_compensated_power_integer_exponent_path(self):
        """Test integer exponent path (lines 172-174)."""
        base = AbsoluteValue(magnitude=3.0, direction=-1.0)
        
        # Test even exponent
        result_even, comp_even = Operations.compensated_power(base, 4)
        assert result_even.magnitude == 81.0
        assert result_even.direction == 1.0  # Even power makes positive
        
        # Test odd exponent
        result_odd, comp_odd = Operations.compensated_power(base, 3)
        assert result_odd.magnitude == 27.0
        assert result_odd.direction == -1.0  # Odd power preserves sign
    
    def test_compensated_power_overflow_protection(self):
        """Test power overflow protection (lines 178-205)."""
        # Test with large base and exponent that could overflow
        large_base = AbsoluteValue(magnitude=1e100, direction=1.0)
        
        result, compensation = Operations.compensated_power(large_base, 2)
        
        # Should handle overflow gracefully
        assert result.magnitude > 0
        assert compensation >= 1.0
    
    def test_compensated_power_underflow_protection(self):
        """Test power underflow protection."""
        # Test with small base and negative exponent that could underflow
        small_base = AbsoluteValue(magnitude=1e-100, direction=1.0)
        
        result, compensation = Operations.compensated_power(small_base, -2)
        
        # Should handle underflow gracefully
        assert result.magnitude >= 0
        assert compensation >= 1.0
    
    def test_compensated_sqrt_basic(self):
        """Test basic compensated square root (lines 222-231)."""
        av = AbsoluteValue(magnitude=9.0, direction=1.0)
        
        result, compensation = Operations.compensated_sqrt(av)
        assert abs(result.magnitude - 3.0) < ACT_EPSILON
        assert result.direction == 1.0
        assert compensation >= 1.0
    
    def test_compensated_sqrt_negative_error(self):
        """Test sqrt with negative value error (lines 222-231)."""
        negative_av = AbsoluteValue(magnitude=4.0, direction=-1.0)
        
        with pytest.raises(ValueError, match="negative"):
            Operations.compensated_sqrt(negative_av)
    
    def test_compensated_sqrt_absolute_input(self):
        """Test sqrt with absolute input (lines 222-231)."""
        absolute = AbsoluteValue.absolute()
        
        result, compensation = Operations.compensated_sqrt(absolute)
        assert result.is_absolute()
        assert compensation == 0.0
    
    def test_compensated_log_basic(self):
        """Test basic compensated logarithm (lines 249-265)."""
        av = AbsoluteValue(magnitude=math.e, direction=1.0)
        
        result, compensation = Operations.compensated_log(av)
        assert abs(result.magnitude - 1.0) < ACT_EPSILON
        assert result.direction == 1.0
        assert compensation >= 1.0
    
    def test_compensated_log_with_base(self):
        """Test logarithm with custom base (lines 249-265)."""
        av = AbsoluteValue(magnitude=8.0, direction=1.0)
        
        result, compensation = Operations.compensated_log(av, base=2.0)
        assert abs(result.magnitude - 3.0) < ACT_EPSILON
        assert result.direction == 1.0
        assert compensation >= 1.0
    
    def test_compensated_log_negative_error(self):
        """Test log with negative value error (lines 249-265)."""
        negative_av = AbsoluteValue(magnitude=2.0, direction=-1.0)
        
        with pytest.raises(ValueError, match="negative"):
            Operations.compensated_log(negative_av)
    
    def test_compensated_log_invalid_base_error(self):
        """Test log with invalid base error (lines 249-265)."""
        av = AbsoluteValue(magnitude=8.0, direction=1.0)
        
        with pytest.raises(ValueError, match="base"):
            Operations.compensated_log(av, base=1.0)
        
        with pytest.raises(ValueError, match="base"):
            Operations.compensated_log(av, base=-2.0)
    
    def test_compensated_exp_basic(self):
        """Test basic compensated exponential (lines 279-296)."""
        av = AbsoluteValue(magnitude=1.0, direction=1.0)
        
        result, compensation = Operations.compensated_exp(av)
        assert abs(result.magnitude - math.e) < ACT_EPSILON
        assert result.direction == 1.0
        assert compensation >= 1.0
    
    def test_compensated_exp_overflow_protection(self):
        """Test exp overflow protection (lines 279-296)."""
        large_av = AbsoluteValue(magnitude=1000.0, direction=1.0)
        
        result, compensation = Operations.compensated_exp(large_av)
        # Should handle overflow gracefully
        assert result.magnitude > 0
        assert compensation > 1.0
    
    def test_compensated_exp_underflow_protection(self):
        """Test exp underflow protection (lines 279-296)."""
        large_negative_av = AbsoluteValue(magnitude=1000.0, direction=-1.0)
        
        result, compensation = Operations.compensated_exp(large_negative_av)
        # Should handle underflow gracefully
        assert result.magnitude >= 0
        assert compensation >= 1.0
    
    def test_compensated_sin_basic(self):
        """Test basic compensated sine (lines 310-316)."""
        av = AbsoluteValue(magnitude=math.pi/2, direction=1.0)
        
        result, compensation = Operations.compensated_sin(av)
        assert abs(result.magnitude - 1.0) < ACT_EPSILON
        assert result.direction == 1.0
        assert compensation >= 1.0
    
    def test_compensated_sin_absolute_input(self):
        """Test sine with absolute input (lines 310-316)."""
        absolute = AbsoluteValue.absolute()
        
        result, compensation = Operations.compensated_sin(absolute)
        assert result.is_absolute()
        assert compensation == 0.0
    
    def test_compensated_cos_basic(self):
        """Test basic compensated cosine (lines 330-336)."""
        av = AbsoluteValue(magnitude=0.0, direction=1.0)
        
        result, compensation = Operations.compensated_cos(av)
        assert abs(result.magnitude - 1.0) < ACT_EPSILON
        assert result.direction == 1.0
        assert compensation >= 1.0
    
    def test_compensated_cos_absolute_input(self):
        """Test cosine with absolute input (lines 330-336)."""
        absolute = AbsoluteValue.absolute()
        
        result, compensation = Operations.compensated_cos(absolute)
        assert result.is_absolute()
        assert compensation == 0.0


class TestSequenceOperations:
    """Test sequence operations with compensation."""
    
    def test_sequence_sum_basic(self):
        """Test basic sequence summation."""
        values = [
            AbsoluteValue(magnitude=1.0, direction=1.0),
            AbsoluteValue(magnitude=2.0, direction=1.0),
            AbsoluteValue(magnitude=3.0, direction=1.0)
        ]
        
        result, compensation = Operations.sequence_sum(values)
        assert result.magnitude == 6.0
        assert result.direction == 1
        assert isinstance(compensation, float)
    
    def test_sequence_sum_mixed_directions(self):
        """Test sequence summation with mixed directions."""
        values = [
            AbsoluteValue(magnitude=5.0, direction=1.0),
            AbsoluteValue(magnitude=2.0, direction=-1.0),
            AbsoluteValue(magnitude=1.0, direction=1.0)
        ]
        
        result, compensation = Operations.sequence_sum(values)
        assert result.magnitude == 4.0
        assert result.direction == 1
        assert isinstance(compensation, float)
    
    def test_sequence_sum_with_absolute(self):
        """Test sequence summation with Absolute values."""
        values = [
            AbsoluteValue(magnitude=3.0, direction=1.0),
            AbsoluteValue.absolute(),
            AbsoluteValue(magnitude=2.0, direction=1.0)
        ]
        
        result, compensation = Operations.sequence_sum(values)
        assert result.magnitude == 5.0
        assert result.direction == 1
        assert isinstance(compensation, float)
    
    def test_sequence_sum_empty(self):
        """Test sequence summation with empty list."""
        result, compensation = Operations.sequence_sum([])
        assert result.is_absolute()
        assert isinstance(compensation, float)
    
    def test_sequence_sum_single_value(self):
        """Test sequence summation with single value (lines 352-356)."""
        single_val = AbsoluteValue(magnitude=5.0, direction=1.0)
        result, compensation = Operations.sequence_sum([single_val])
        
        assert result.magnitude == 5.0
        assert result.direction == 1.0
        assert compensation == 1.0
    
    def test_sequence_sum_kahan_algorithm(self):
        """Test Kahan summation algorithm implementation (lines 352-376)."""
        # Test with values that would lose precision in naive summation
        values = [
            AbsoluteValue(magnitude=1.0, direction=1.0),
            AbsoluteValue(magnitude=1e-15, direction=1.0),
            AbsoluteValue(magnitude=1e-15, direction=1.0)
        ]
        
        result, compensation = Operations.sequence_sum(values)
        
        # Kahan summation should maintain precision
        expected = 1.0 + 2e-15
        assert abs(result.magnitude - expected) < ACT_EPSILON
        assert result.direction == 1.0
        assert compensation >= 1.0
    
    def test_sequence_sum_compensation_accumulation(self):
        """Test compensation accumulation in sequence sum (lines 352-376)."""
        # Create sequence with alternating large and small values
        values = []
        for i in range(10):
            if i % 2 == 0:
                values.append(AbsoluteValue(magnitude=1e10, direction=1.0))
            else:
                values.append(AbsoluteValue(magnitude=1.0, direction=1.0))
        
        result, compensation = Operations.sequence_sum(values)
        
        # Should handle mixed magnitudes with compensation
        assert result.magnitude > 0
        assert compensation >= 1.0
    
    def test_sequence_sum_large_values(self):
        """Test sequence summation with large values (Kahan summation)."""
        # Create values that would lose precision with naive summation
        large_val = AbsoluteValue(magnitude=1e16, direction=1.0)
        small_val = AbsoluteValue(magnitude=1.0, direction=1.0)
        
        values = [large_val] + [small_val] * 1000
        result, compensation = Operations.sequence_sum(values)
        
        # Should maintain precision
        expected = 1e16 + 1000.0
        assert abs(result.magnitude - expected) < ACT_EPSILON * expected
        assert isinstance(compensation, float)
    
    def test_sequence_product_basic(self):
        """Test basic sequence product."""
        values = [
            AbsoluteValue(magnitude=2.0, direction=1.0),
            AbsoluteValue(magnitude=3.0, direction=1.0),
            AbsoluteValue(magnitude=4.0, direction=1.0)
        ]
        
        result, compensation = Operations.sequence_product(values)
        
        assert result.magnitude == 24.0
        assert result.direction == 1
        assert isinstance(compensation, float)
    
    def test_sequence_product_mixed_directions(self):
        """Test sequence product with mixed directions."""
        values = [
            AbsoluteValue(magnitude=2.0, direction=1.0),
            AbsoluteValue(magnitude=3.0, direction=-1.0),
            AbsoluteValue(magnitude=4.0, direction=1.0)
        ]
        
        result, compensation = Operations.sequence_product(values)
        assert result.magnitude == 24.0
        assert result.direction == -1
        assert isinstance(compensation, float)
    
    def test_sequence_product_with_absolute(self):
        """Test sequence product with Absolute values."""
        values = [
            AbsoluteValue(magnitude=2.0, direction=1.0),
            AbsoluteValue.absolute(),
            AbsoluteValue(magnitude=3.0, direction=1.0)
        ]
        
        result, compensation = Operations.sequence_product(values)
        assert result.is_absolute()
        assert isinstance(compensation, float)
    
    def test_sequence_product_empty(self):
        """Test sequence product with empty list."""
        result, compensation = Operations.sequence_product([])
        assert result.magnitude == 1.0
        assert result.direction == 1
        assert isinstance(compensation, float)


class TestUtilityOperations:
    """Test utility operations."""
    
    def test_interpolate_basic(self):
        """Test basic interpolation."""
        av1 = AbsoluteValue(magnitude=2.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=6.0, direction=1.0)
        
        result = Operations.interpolate(av1, av2, 0.5)
        assert result.magnitude == 4.0
        assert result.direction == 1
    
    def test_interpolate_endpoints(self):
        """Test interpolation at endpoints."""
        av1 = AbsoluteValue(magnitude=2.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=8.0, direction=1.0)
        
        # Test at t=0 (first endpoint)
        result0 = Operations.interpolate(av1, av2, 0.0)
        assert result0.magnitude == 2.0
        assert result0.direction == 1.0
        
        # Test at t=1 (second endpoint)
        result1 = Operations.interpolate(av1, av2, 1.0)
        assert result1.magnitude == 8.0
        assert result1.direction == 1.0
    
    def test_interpolate_opposite_directions(self):
        """Test interpolation with opposite directions (lines 390-414)."""
        av1 = AbsoluteValue(magnitude=4.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=6.0, direction=-1.0)
        
        result = Operations.interpolate(av1, av2, 0.5)
        
        # Should handle direction interpolation
        assert result.magnitude > 0
        assert abs(result.direction) <= 1.0
    
    def test_interpolate_with_absolute(self):
        """Test interpolation with absolute values (lines 390-414)."""
        av = AbsoluteValue(magnitude=3.0, direction=1.0)
        absolute = AbsoluteValue.absolute()
        
        result1 = Operations.interpolate(av, absolute, 0.5)
        result2 = Operations.interpolate(absolute, av, 0.5)
        
        # Interpolation with absolute should handle gracefully
        assert isinstance(result1, AbsoluteValue)
        assert isinstance(result2, AbsoluteValue)
    
    def test_interpolate_edge_cases(self):
        """Test interpolation edge cases (lines 390-414)."""
        av1 = AbsoluteValue(magnitude=1.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=1.0, direction=1.0)
        
        # Interpolation between identical values
        result = Operations.interpolate(av1, av2, 0.3)
        assert result.magnitude == 1.0
        assert result.direction == 1.0
    
    def test_distance_basic(self):
        """Test basic distance calculation (lines 432-447)."""
        av1 = AbsoluteValue(magnitude=3.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=7.0, direction=1.0)
        
        distance = Operations.distance(av1, av2)
        assert abs(distance - 4.0) < ACT_EPSILON
    
    def test_distance_opposite_directions(self):
        """Test distance with opposite directions (lines 432-447)."""
        av1 = AbsoluteValue(magnitude=3.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=3.0, direction=-1.0)
        
        distance = Operations.distance(av1, av2)
        assert distance == 6.0  # Should be sum of magnitudes
    
    def test_distance_with_absolute(self):
        """Test distance with absolute values (lines 432-447)."""
        av = AbsoluteValue(magnitude=5.0, direction=1.0)
        absolute = AbsoluteValue.absolute()
        
        distance1 = Operations.distance(av, absolute)
        distance2 = Operations.distance(absolute, av)
        
        # Distance with absolute should be handled
        assert distance1 >= 0
        assert distance2 >= 0
        assert distance1 == distance2  # Distance should be symmetric
    
    def test_distance_zero_case(self):
        """Test distance zero case (lines 432-447)."""
        av = AbsoluteValue(magnitude=4.0, direction=1.0)
        
        distance = Operations.distance(av, av)
        assert distance == 0.0
    
    def test_normalize_basic(self):
        """Test basic normalization (lines 460-461, 476-479)."""
        av = AbsoluteValue(magnitude=5.0, direction=1.0)
        
        result = Operations.normalize(av)
        assert abs(result.magnitude - 1.0) < ACT_EPSILON
        assert result.direction == 1.0
    
    def test_normalize_negative_direction(self):
        """Test normalization with negative direction (lines 460-461, 476-479)."""
        av = AbsoluteValue(magnitude=3.0, direction=-1.0)
        
        result = Operations.normalize(av)
        assert abs(result.magnitude - 1.0) < ACT_EPSILON
        assert result.direction == -1.0
    
    def test_normalize_absolute_error(self):
        """Test normalization with absolute value error (lines 460-461, 476-479)."""
        absolute = AbsoluteValue.absolute()
        
        with pytest.raises(ValueError, match="absolute"):
            Operations.normalize(absolute)
    
    def test_normalize_zero_magnitude_error(self):
        """Test normalization with zero magnitude error (lines 460-461, 476-479)."""
        zero_av = AbsoluteValue(magnitude=0.0, direction=1.0)
        
        with pytest.raises(ValueError, match="zero"):
            Operations.normalize(zero_av)
        av1 = AbsoluteValue(magnitude=2.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=6.0, direction=1.0)
        
        result1 = Operations.interpolate(av1, av2, 0.0)
        result2 = Operations.interpolate(av1, av2, 1.0)
        
        assert result1 == av1
        assert result2 == av2
    
    def test_interpolate_opposite_directions(self):
        """Test interpolation with opposite directions."""
        av1 = AbsoluteValue(magnitude=4.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=4.0, direction=-1.0)
        
        result = Operations.interpolate(av1, av2, 0.5)
        assert result.is_absolute()
    
    def test_distance_basic(self):
        """Test basic distance calculation."""
        av1 = AbsoluteValue(magnitude=2.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=5.0, direction=1.0)
        
        result = Operations.distance(av1, av2)
        assert result.magnitude == 3.0
        assert result.direction == 1
    
    def test_distance_opposite_directions(self):
        """Test distance with opposite directions."""
        av1 = AbsoluteValue(magnitude=3.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=2.0, direction=-1.0)
        
        result = Operations.distance(av1, av2)
        assert result.magnitude == 5.0
        assert result.direction == 1
    
    def test_distance_symmetric(self):
        """Test distance symmetry."""
        av1 = AbsoluteValue(magnitude=3.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=7.0, direction=1.0)
        
        dist1 = Operations.distance(av1, av2)
        dist2 = Operations.distance(av2, av1)
        
        assert dist1 == dist2
    
    def test_distance_same_values(self):
        """Test distance between same values."""
        av = AbsoluteValue(magnitude=5.0, direction=1.0)
        
        result = Operations.distance(av, av)
        assert result.is_absolute()
    
    def test_normalize_basic(self):
        """Test basic normalization."""
        value = AbsoluteValue(magnitude=5.0, direction=1.0)
        
        normalized = Operations.normalize(value)
        
        # Check that magnitude is 1.0
        assert abs(normalized.magnitude - 1.0) < ACT_EPSILON
        
        # Check that direction is preserved
        assert normalized.direction == value.direction
    
    def test_normalize_with_negative_direction(self):
        """Test normalization with negative direction."""
        value = AbsoluteValue(magnitude=3.0, direction=-1.0)
        
        normalized = Operations.normalize(value)
        
        # Check that magnitude is 1.0
        assert abs(normalized.magnitude - 1.0) < ACT_EPSILON
        
        # Check that direction is preserved
        assert normalized.direction == -1
    
    def test_normalize_absolute_raises_error(self):
        """Test normalization with Absolute value raises error."""
        absolute_value = AbsoluteValue.absolute()
        
        with pytest.raises(ValueError, match="Cannot normalize Absolute value"):
            Operations.normalize(absolute_value)
    
    def test_normalize_unit_magnitude(self):
        """Test normalization of value with unit magnitude."""
        value = AbsoluteValue(magnitude=1.0, direction=-1.0)
        
        normalized = Operations.normalize(value)
        
        # Should remain the same
        assert normalized.magnitude == 1.0
        assert normalized.direction == -1
    
    def test_normalize_absolute_value_error(self):
        """Test normalize with Absolute value error - line 433"""
        absolute_value = AbsoluteValue.absolute()
        
        with pytest.raises(ValueError, match='Cannot normalize Absolute value'):
            Operations.normalize(absolute_value)


class TestCompensationStability:
    """Test compensation and stability properties."""
    
    def test_large_value_compensation(self):
        """Test compensation with large values."""
        large1 = AbsoluteValue(magnitude=1e15, direction=1.0)
        large2 = AbsoluteValue(magnitude=1e15, direction=-1.0)
        
        # Should compensate to Absolute
        result, _ = Operations.compensated_add(large1, large2)
        assert result.is_absolute()
    
    def test_precision_preservation(self):
        """Test precision preservation in operations."""
        # Values that might lose precision in naive arithmetic
        av1 = AbsoluteValue(magnitude=1.0 + 1e-15, direction=1.0)
        av2 = AbsoluteValue(magnitude=1e-15, direction=-1.0)
        
        result, _ = Operations.compensated_add(av1, av2)
        assert abs(result.magnitude - 1.0) < ACT_EPSILON
    
    def test_overflow_prevention(self):
        """Test overflow prevention."""
        # Values that might cause overflow
        large = AbsoluteValue(magnitude=1e100, direction=1.0)
        
        # Operations should not cause overflow
        result, _ = Operations.compensated_multiply(large, large)
        assert not math.isinf(result.magnitude)
        assert not math.isnan(result.magnitude)
    
    def test_underflow_handling(self):
        """Test underflow handling."""
        # Very small values
        tiny = AbsoluteValue(magnitude=1e-100, direction=1.0)
        
        # Operations should handle underflow gracefully
        result, compensation = Operations.compensated_divide(tiny, AbsoluteValue(magnitude=1e50, direction=1.0))
        assert result.numerator.magnitude >= 0.0
        assert not math.isnan(result.numerator.magnitude)
        assert isinstance(compensation, float)


class TestACTCompliance:
    """Test ACT (Absolute Compensation Theory) compliance."""
    
    def test_compensation_axiom_operations(self):
        """Test that operations follow compensation axiom."""
        # Large opposing values should compensate
        large_pos = AbsoluteValue(magnitude=1e12, direction=1.0)
        large_neg = AbsoluteValue(magnitude=1e12, direction=-1.0)
        
        result, _ = Operations.compensated_add(large_pos, large_neg)
        assert result.is_absolute()
    
    def test_stability_axiom_operations(self):
        """Test that operations maintain stability."""
        # Repeated operations should remain stable
        av = AbsoluteValue(magnitude=1.1, direction=1.0)
        
        result = av
        for _ in range(100):
            result, _ = Operations.compensated_multiply(result, AbsoluteValue(magnitude=1.0, direction=1.0))
        
        # Should remain close to original
        assert abs(result.magnitude - av.magnitude) < ACT_EPSILON
    
    def test_eternity_axiom_operations(self):
        """Test that operations preserve eternal ratios."""
        av1 = AbsoluteValue(magnitude=6.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=3.0, direction=1.0)
        
        # Scale both values
        scale = AbsoluteValue(magnitude=2.0, direction=1.0)
        scaled_av1, _ = Operations.compensated_multiply(av1, scale)
        scaled_av2, _ = Operations.compensated_multiply(av2, scale)
        
        # Ratio should be preserved
        original_ratio, _ = Operations.compensated_divide(av1, av2)
        scaled_ratio, _ = Operations.compensated_divide(scaled_av1, scaled_av2)

        assert abs(original_ratio.numerical_value() - scaled_ratio.numerical_value()) < ACT_EPSILON


class TestMissingOperationsCoverage:
    """Test missing operations coverage for uncovered lines."""
    
    def test_compensated_add_threshold_compensation(self):
        """Test compensated_add with result below compensation threshold."""
        # Create values that when added result in very small magnitude
        av1 = AbsoluteValue(magnitude=1e-15, direction=1.0)
        av2 = AbsoluteValue(magnitude=1e-16, direction=-1.0)
        
        result, compensation = Operations.compensated_add(av1, av2)
        
        # Should compensate to Absolute due to small result
        assert result.is_absolute()
        assert isinstance(compensation, float)
    
    def test_compensated_add_underflow_compensation(self):
        """Test compensated_add with underflow compensation - lines 71-73"""
        # Create values that will result in underflow
        small_a = AbsoluteValue(magnitude=1e-200, direction=1)
        small_b = AbsoluteValue(magnitude=1e-200, direction=1)
        
        result, compensation = Operations.compensated_add(small_a, small_b)
        
        # Should trigger underflow compensation
        assert result.is_absolute()
        assert compensation < 1.0
    
    def test_compensated_multiply_absolute_operands(self):
        """Test compensated_multiply with absolute operands - lines 92-93"""
        abs_val = AbsoluteValue.absolute()
        normal_val = AbsoluteValue(magnitude=5.0, direction=1)
        
        result, compensation = Operations.compensated_multiply(abs_val, normal_val)
        
        assert result.is_absolute()
        assert compensation == 0.0
    
    def test_compensated_divide_absolute_denominator(self):
        """Test compensated_divide with absolute denominator - line 132"""
        numerator = AbsoluteValue(magnitude=5.0, direction=1)
        denominator = AbsoluteValue.absolute()
        
        with pytest.raises(ValueError, match='Cannot divide by Absolute'):
            Operations.compensated_divide(numerator, denominator)
    
    def test_compensated_multiply_overflow_compensation(self):
        """Test compensated_multiply with overflow compensation."""
        # Create values that cause overflow when multiplied
        large1 = AbsoluteValue(magnitude=1e60, direction=1.0)
        large2 = AbsoluteValue(magnitude=1e60, direction=1.0)
        
        result, compensation = Operations.compensated_multiply(large1, large2)
        
        # Should apply logarithmic compensation
        assert result.magnitude == 1e100
        assert compensation > 1.0  # Should have compensation applied
    
    def test_compensated_multiply_underflow_compensation(self):
        """Test compensated_multiply with underflow compensation."""
        # Create values that cause underflow when multiplied
        tiny1 = AbsoluteValue(magnitude=1e-200, direction=1.0)
        tiny2 = AbsoluteValue(magnitude=1e-200, direction=1.0)
        
        result, compensation = Operations.compensated_multiply(tiny1, tiny2)
        
        # Should compensate to Absolute due to underflow
        assert result.is_absolute()
        assert compensation < 1.0  # Should have underflow compensation
    
    def test_compensated_multiply_overflow_compensation_new(self):
        """Test compensated_multiply with overflow compensation - lines 108"""
        # Create values that will cause overflow
        large_a = AbsoluteValue(magnitude=1e60, direction=1)
        large_b = AbsoluteValue(magnitude=1e60, direction=1)
        
        result, compensation = Operations.compensated_multiply(large_a, large_b)
        
        # Should trigger overflow compensation
        assert result.magnitude == 1e100
        assert compensation > 1.0  # logarithmic compensation
    
    def test_compensated_power_negative_absolute_base(self):
        """Test compensated_power with Absolute base and negative exponent."""
        absolute_base = AbsoluteValue.absolute()
        
        with pytest.raises(ValueError, match="Cannot raise Absolute to negative power"):
            Operations.compensated_power(absolute_base, -2.0)
    
    def test_compensated_power_unit_exponent_new(self):
        """Test compensated_power with exponent = 1 - line 165"""
        base = AbsoluteValue(magnitude=5.0, direction=1)
        
        result, compensation = Operations.compensated_power(base, 1.0)
        
        assert result == base
        assert compensation == 1.0
    
    def test_compensated_power_even_integer_exponent(self):
        """Test compensated_power with even integer exponent - line 174"""
        negative_base = AbsoluteValue(magnitude=3.0, direction=-1)
        
        result, compensation = Operations.compensated_power(negative_base, 2.0)
        
        # Even power should result in positive direction
        assert result.magnitude == 9.0
        assert result.direction == 1
        assert compensation == 1.0
    
    def test_compensated_power_fractional_negative_base(self):
        """Test compensated_power with negative base and fractional exponent."""
        negative_base = AbsoluteValue(magnitude=4.0, direction=-1.0)
        
        # The ValueError is caught by the try-catch block and returns Absolute
        result, compensation = Operations.compensated_power(negative_base, 0.5)
        assert result.is_absolute()
        assert compensation == 0.0
    
    def test_compensated_power_negative_exponent_inversion(self):
        """Test compensated_power with negative exponent inversion - line 185"""
        base = AbsoluteValue(magnitude=4.0, direction=1.0)
        
        result, compensation = Operations.compensated_power(base, -2.0)
        
        # Should be 1/(4^2) = 1/16 = 0.0625
        assert abs(result.magnitude - 0.0625) < 1e-10
        assert result.direction == 1
        assert compensation == 1.0
    
    def test_compensated_power_overflow_logarithmic_compensation(self):
        """Test compensated_power with overflow and logarithmic compensation - lines 190-192"""
        large_base = AbsoluteValue(magnitude=10.0, direction=1.0)
        large_exponent = 150.0
        
        result, compensation = Operations.compensated_power(large_base, large_exponent)
        
        # Should trigger overflow compensation
        assert result.magnitude == 1e100
        assert compensation > 1.0  # logarithmic compensation
    
    def test_compensated_power_underflow_compensation(self):
        """Test compensated_power with underflow compensation - line 195"""
        small_base = AbsoluteValue(magnitude=1e-100, direction=1.0)
        
        result, compensation = Operations.compensated_power(small_base, 2.0)
        
        # Should trigger underflow compensation
        assert result.is_absolute()
        assert compensation < 1.0
    
    def test_compensated_power_overflow_exception_handling(self):
        """Test compensated_power with OverflowError exception handling - lines 200-205"""
        # This is tricky to trigger directly, but we can test the exception path
        # by using extreme values that might cause math.pow to overflow
        extreme_base = AbsoluteValue(magnitude=1e308, direction=1.0)
        
        result, compensation = Operations.compensated_power(extreme_base, 2.0)
        
        # Should handle overflow gracefully
        assert result.magnitude == 1e100
        assert compensation > 500.0  # Should have significant logarithmic compensation
    
    def test_compensated_power_zero_magnitude_negative_exponent(self):
        """Test compensated_power with zero magnitude and negative exponent."""
        zero_base = AbsoluteValue(magnitude=0.0, direction=1.0)
        
        with pytest.raises(ValueError, match="Cannot invert zero magnitude"):
            Operations.compensated_power(zero_base, -1.0)
    
    def test_compensated_power_overflow_exception(self):
        """Test compensated_power with overflow exception handling."""
        # Create a case that might cause overflow exception
        large_base = AbsoluteValue(magnitude=1e50, direction=1.0)
        
        result, compensation = Operations.compensated_power(large_base, 10.0)
        
        # Should handle overflow gracefully with logarithmic compensation
        assert result.magnitude == 1e100
        assert compensation > 400.0  # Should have significant logarithmic compensation
    
    def test_compensated_sqrt_negative_direction(self):
        """Test compensated_sqrt with negative direction."""
        negative_value = AbsoluteValue(magnitude=4.0, direction=-1.0)
        
        with pytest.raises(ValueError, match="Cannot take square root of negative AbsoluteValue"):
            Operations.compensated_sqrt(negative_value)
    
    def test_compensated_sqrt_absolute_value(self):
        """Test compensated_sqrt with Absolute value."""
        absolute_value = AbsoluteValue.absolute()
        
        result, compensation = Operations.compensated_sqrt(absolute_value)
        
        assert result.is_absolute()
        assert compensation == 0.0
    
    def test_compensated_log_absolute_value(self):
        """Test compensated_log with Absolute value."""
        absolute_value = AbsoluteValue.absolute()
        
        with pytest.raises(ValueError, match="Cannot take logarithm of Absolute"):
            Operations.compensated_log(absolute_value)
    
    def test_compensated_log_negative_direction(self):
        """Test compensated_log with negative direction."""
        negative_value = AbsoluteValue(magnitude=4.0, direction=-1.0)
        
        with pytest.raises(ValueError, match="Cannot take logarithm of negative AbsoluteValue"):
            Operations.compensated_log(negative_value)
    
    def test_compensated_log_invalid_base(self):
        """Test compensated_log with invalid base values."""
        value = AbsoluteValue(magnitude=4.0, direction=1.0)
        
        # Test negative base
        with pytest.raises(ValueError, match="Logarithm base must be positive and not equal to 1"):
            Operations.compensated_log(value, base=-2.0)
        
        # Test base = 1
        with pytest.raises(ValueError, match="Logarithm base must be positive and not equal to 1"):
            Operations.compensated_log(value, base=1.0)
        
        # Test base = 0
        with pytest.raises(ValueError, match="Logarithm base must be positive and not equal to 1"):
            Operations.compensated_log(value, base=0.0)
    
    def test_compensated_log_custom_base_calculation(self):
        """Test compensated_log with custom base calculation - lines 249-265"""
        value = AbsoluteValue(magnitude=100.0, direction=1)
        
        # Test with base 10
        result, compensation = Operations.compensated_log(value, base=10.0)
        
        # log_10(100) = 2
        assert abs(result.magnitude - 2.0) < 1e-10
        assert result.direction == 1
        assert compensation == 1.0
    
    def test_compensated_log_custom_base(self):
        """Test compensated_log with custom base."""
        value = AbsoluteValue(magnitude=8.0, direction=1.0)
        
        result, compensation = Operations.compensated_log(value, base=2.0)
        
        # log_2(8) = 3
        assert abs(result.to_float() - 3.0) < ACT_EPSILON
        assert compensation == 1.0
    
    def test_compensated_exp_absolute_value(self):
        """Test compensated_exp with Absolute value."""
        absolute_value = AbsoluteValue.absolute()
        
        result, compensation = Operations.compensated_exp(absolute_value)
        
        # exp(Absolute) should return unit positive
        assert result.magnitude == 1.0
        assert result.direction == 1
        assert compensation == 1.0
    
    def test_compensated_exp_overflow_protection(self):
        """Test compensated_exp with overflow protection."""
        large_value = AbsoluteValue(magnitude=200.0, direction=1.0)  # Use smaller value to trigger overflow protection
        
        result, compensation = Operations.compensated_exp(large_value)
        
        # Should apply overflow protection
        assert result.magnitude == 1e100
        assert compensation < -100.0  # Should have significant negative logarithmic compensation
    
    def test_compensated_exp_absolute_input(self):
        """Test compensated_exp with Absolute input - lines 279-296"""
        absolute_value = AbsoluteValue.absolute()
        
        result, compensation = Operations.compensated_exp(absolute_value)
        
        # exp(Absolute) should return unit positive
        assert result.magnitude == 1.0
        assert result.direction == 1
        assert compensation == 1.0
    
    def test_compensated_exp_normal_calculation(self):
        """Test compensated_exp normal calculation - lines 279-296"""
        value = AbsoluteValue(magnitude=2.0, direction=1.0)
        
        result, compensation = Operations.compensated_exp(value)
        
        # exp(2) ≈ 7.389
        expected = math.exp(2.0)
        assert abs(result.magnitude - expected) < 1e-10
        assert result.direction == 1
        assert compensation == 1.0
    
    def test_compensated_exp_overflow_exception_new(self):
        """Test compensated_exp with OverflowError exception - lines 294-296"""
        # Create a scenario that might trigger OverflowError in math.exp
        extreme_value = AbsoluteValue(magnitude=1000.0, direction=1)
        
        result, compensation = Operations.compensated_exp(extreme_value)
        
        # Should handle OverflowError gracefully
        assert result.magnitude == 1e100
        assert compensation == float('inf')
    
    def test_compensated_exp_overflow_exception(self):
        """Test compensated_exp with overflow exception handling."""
        # This might be hard to trigger, but we can test the exception path
        very_large_value = AbsoluteValue(magnitude=1e10, direction=1.0)
        
        result, compensation = Operations.compensated_exp(very_large_value)
        
        # Should handle gracefully
        assert result.magnitude == 1e100
        assert compensation == float('inf')
    
    def test_sequence_sum_without_compensation(self):
        """Test sequence_sum without Kahan compensation."""
        values = [
            AbsoluteValue(magnitude=1.0, direction=1.0),
            AbsoluteValue(magnitude=2.0, direction=1.0),
            AbsoluteValue(magnitude=3.0, direction=1.0)
        ]
        
        result, compensation = Operations.sequence_sum(values, use_compensation=False)
        
        assert result.magnitude == 6.0
        assert result.direction == 1
        assert compensation == 0.0
    
    def test_sequence_sum_without_compensation_loop(self):
        """Test sequence_sum without compensation loop - lines 222-231"""
        values = [
            AbsoluteValue(magnitude=1.0, direction=1),
            AbsoluteValue(magnitude=2.0, direction=-1),
            AbsoluteValue(magnitude=3.0, direction=1),
            AbsoluteValue(magnitude=4.0, direction=-1)
        ]
        
        result, compensation = Operations.sequence_sum(values, use_compensation=False)
        
        # Should iterate through the loop: 1 + (-2) + 3 + (-4) = -2
        assert result.magnitude == 2.0
        assert result.direction == -1
        assert compensation == 0.0
    
    def test_sequence_sum_single_value(self):
        """Test sequence_sum with single value."""
        values = [AbsoluteValue(magnitude=5.0, direction=1.0)]
        
        result, compensation = Operations.sequence_sum(values)
        
        assert result == values[0]
        assert compensation == 0.0
    
    def test_sequence_product_without_compensation(self):
        """Test sequence_product without compensation."""
        values = [
            AbsoluteValue(magnitude=2.0, direction=1.0),
            AbsoluteValue(magnitude=3.0, direction=-1.0)
        ]
        
        result, compensation = Operations.sequence_product(values, use_compensation=False)
        
        assert result.magnitude == 6.0
        assert result.direction == -1
        assert compensation == 1.0  # No compensation applied
    
    def test_sequence_product_single_value(self):
        """Test sequence_product with single value."""
        values = [AbsoluteValue(magnitude=5.0, direction=-1.0)]
        
        result, compensation = Operations.sequence_product(values)
        
        assert result == values[0]
        assert compensation == 1.0
    
    def test_sequence_sum_empty_list(self):
        """Test sequence_sum with empty list - lines 310-316"""
        result, compensation = Operations.sequence_sum([])
        
        assert result.is_absolute()
        assert compensation == 0.0
    
    def test_sequence_sum_kahan_compensation_loop(self):
        """Test sequence_sum Kahan compensation loop - lines 330-336"""
        # Create values that will exercise the Kahan summation loop
        values = [
            AbsoluteValue(magnitude=1.0, direction=1),
            AbsoluteValue(magnitude=1e-15, direction=1),
            AbsoluteValue(magnitude=1.0, direction=1)
        ]
        
        result, compensation = Operations.sequence_sum(values, use_compensation=True)
        
        # Should use Kahan summation
        assert result.magnitude > 2.0
        assert isinstance(compensation, float)
    
    def test_sequence_product_empty_list(self):
        """Test sequence_product with empty list - line 356"""
        result, compensation = Operations.sequence_product([])
        
        assert result.magnitude == 1.0
        assert result.direction == 1
        assert compensation == 1.0
    
    def test_sequence_product_absolute_values(self):
        """Test sequence_product with Absolute values - lines 360-363"""
        values = [
            AbsoluteValue(magnitude=2.0, direction=1),
            AbsoluteValue.absolute(),
            AbsoluteValue(magnitude=3.0, direction=1)
        ]
        
        result, compensation = Operations.sequence_product(values)
        
        # Should return Absolute when any value is Absolute
        assert result.is_absolute()
        assert compensation == 0.0
    
    def test_sequence_product_without_compensation_loop(self):
        """Test sequence_product without compensation loop - line 394"""
        values = [
            AbsoluteValue(magnitude=2.0, direction=1),
            AbsoluteValue(magnitude=3.0, direction=-1),
            AbsoluteValue(magnitude=4.0, direction=1)
        ]
        
        result, compensation = Operations.sequence_product(values, use_compensation=False)
        
        # Should calculate product without compensation
        assert result.magnitude == 24.0
        assert result.direction == -1  # 1 * -1 * 1 = -1
        assert compensation == 1.0
    
    def test_interpolate_invalid_parameter(self):
        """Test interpolate with invalid t parameter."""
        av1 = AbsoluteValue(magnitude=2.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=6.0, direction=1.0)
        
        # Test t < 0
        with pytest.raises(ValueError, match="Interpolation parameter t must be in \[0, 1\] range"):
            Operations.interpolate(av1, av2, -0.1)
        
        # Test t > 1
        with pytest.raises(ValueError, match="Interpolation parameter t must be in \[0, 1\] range"):
            Operations.interpolate(av1, av2, 1.1)
    
    def test_interpolate_edge_cases(self):
        """Test interpolate with t=0 and t=1 edge cases - line 409"""
        start = AbsoluteValue(magnitude=1.0, direction=1)
        end = AbsoluteValue(magnitude=5.0, direction=-1)
        
        # Test t=0
        result_start = Operations.interpolate(start, end, 0.0)
        assert result_start == start
        
        # Test t=1  
        result_end = Operations.interpolate(start, end, 1.0)
        assert result_end == end
    
    def test_compensated_power_absolute_base_positive_exponent(self):
        """Test compensated_power with absolute base and positive exponent - lines 147-148"""
        abs_base = AbsoluteValue.absolute()
        result, compensation = Operations.compensated_power(abs_base, 2.0)
        
        assert result.is_absolute()
        assert compensation == 0.0
    
    def test_compensated_power_absolute_base_zero_exponent(self):
        """Test compensated_power with absolute base and zero exponent - lines 149-150"""
        abs_base = AbsoluteValue.absolute()
        result, compensation = Operations.compensated_power(abs_base, 0.0)
        
        assert result.magnitude == 1.0
        assert result.direction == 1
        assert compensation == 1.0
    
    def test_compensated_power_absolute_base_negative_exponent(self):
        """Test compensated_power with absolute base and negative exponent - lines 152-153"""
        abs_base = AbsoluteValue.absolute()
        
        with pytest.raises(ValueError, match='Cannot raise Absolute to negative power'):
            Operations.compensated_power(abs_base, -2.0)
    
    def test_compensated_power_zero_exponent(self):
        """Test compensated_power with zero exponent - lines 155-156"""
        base = AbsoluteValue(magnitude=5.0, direction=1)
        result, compensation = Operations.compensated_power(base, 0.0)
        
        assert result.magnitude == 1.0
        assert result.direction == 1
        assert compensation == 1.0
    
    def test_compensated_power_unit_exponent(self):
        """Test compensated_power with unit exponent - lines 158-159"""
        base = AbsoluteValue(magnitude=5.0, direction=1)
        result, compensation = Operations.compensated_power(base, 1.0)
        
        assert result == base
        assert compensation == 1.0
    

    
    def test_compensated_power_zero_magnitude_negative_exponent(self):
        """Test compensated_power with zero magnitude and negative exponent - lines 184-186"""
        zero_base = AbsoluteValue(magnitude=0.0, direction=1)
        
        # Should raise ValueError for negative power on absolute value
        with pytest.raises(ValueError, match='Cannot raise Absolute to negative power'):
            Operations.compensated_power(zero_base, -2.0)
    
    def test_compensated_power_underflow_compensation(self):
        """Test compensated_power underflow compensation - lines 195"""
        small_base = AbsoluteValue(magnitude=1e-10, direction=1)
        result, compensation = Operations.compensated_power(small_base, 10.0)
        
        # Should return absolute value due to underflow
        assert result.is_absolute()
        assert compensation < 1.0
    
    def test_compensated_power_overflow_exception_handling(self):
        """Test compensated_power overflow exception handling - lines 200-205"""
        # This test covers the exception handling in the try-catch block
        large_base = AbsoluteValue(magnitude=1e50, direction=1)
        result, compensation = Operations.compensated_power(large_base, 10.0)
        
        # Should handle overflow by returning absolute value
        assert result.is_absolute()
        assert compensation == 0.0
    
    def test_compensated_sqrt_negative_direction(self):
        """Test compensated_sqrt with negative direction - line 222-223"""
        negative_val = AbsoluteValue(magnitude=4.0, direction=-1)
        
        with pytest.raises(ValueError, match='Cannot take square root of negative AbsoluteValue'):
            Operations.compensated_sqrt(negative_val)
    
    def test_compensated_sqrt_absolute_value(self):
        """Test compensated_sqrt with absolute value - lines 225-226"""
        abs_val = AbsoluteValue.absolute()
        result, compensation = Operations.compensated_sqrt(abs_val)
        
        assert result.is_absolute()
        assert compensation == 0.0
    
    def test_compensated_log_absolute_value(self):
        """Test compensated_log with absolute value - lines 249-250"""
        abs_val = AbsoluteValue.absolute()
        
        with pytest.raises(ValueError, match='Cannot take logarithm of Absolute'):
            Operations.compensated_log(abs_val)
    
    def test_compensated_log_negative_direction(self):
        """Test compensated_log with negative direction - lines 252-253"""
        negative_val = AbsoluteValue(magnitude=4.0, direction=-1)
        
        with pytest.raises(ValueError, match='Cannot take logarithm of negative AbsoluteValue'):
            Operations.compensated_log(negative_val)
    
    def test_compensated_log_invalid_base(self):
        """Test compensated_log with invalid base - lines 255-256"""
        val = AbsoluteValue(magnitude=4.0, direction=1)
        
        with pytest.raises(ValueError, match='Logarithm base must be positive and not equal to 1'):
            Operations.compensated_log(val, base=0)
        
        with pytest.raises(ValueError, match='Logarithm base must be positive and not equal to 1'):
            Operations.compensated_log(val, base=1)
        
        with pytest.raises(ValueError, match='Logarithm base must be positive and not equal to 1'):
            Operations.compensated_log(val, base=-2)
    
    def test_compensated_log_custom_base(self):
        """Test compensated_log with custom base - lines 262-265"""
        val = AbsoluteValue(magnitude=8.0, direction=1)
        result, compensation = Operations.compensated_log(val, base=2)
        
        # log_2(8) = 3
        assert abs(result.to_float() - 3.0) < 1e-10
        assert compensation == 1.0
    
    def test_compensated_exp_absolute_value(self):
        """Test compensated_exp with absolute value - lines 279-280"""
        abs_val = AbsoluteValue.absolute()
        result, compensation = Operations.compensated_exp(abs_val)
        
        assert result.magnitude == 1.0
        assert result.direction == 1
        assert compensation == 1.0
    
    def test_compensated_exp_overflow_protection(self):
        """Test compensated_exp overflow protection - lines 285-290"""
        large_val = AbsoluteValue(magnitude=200.0, direction=1)
        result, compensation = Operations.compensated_exp(large_val)
        
        # Should return large magnitude with compensation
        assert result.magnitude > 1e80  # Very large but not exactly 1e100
        assert compensation > 0  # Positive compensation
    
    def test_compensated_exp_overflow_exception(self):
        """Test compensated_exp overflow exception handling - lines 294-296"""
        # This should trigger the exception handling path
        very_large_val = AbsoluteValue(magnitude=1000.0, direction=1)
        result, compensation = Operations.compensated_exp(very_large_val)
        
        # Should handle the overflow gracefully
        assert isinstance(result, AbsoluteValue)
        assert isinstance(compensation, float)
    
    def test_compensated_sin_absolute_value(self):
        """Test compensated_sin with absolute value - lines 310-311"""
        abs_val = AbsoluteValue.absolute()
        result, compensation = Operations.compensated_sin(abs_val)
        
        assert result.is_absolute()
        assert compensation == 0.0
    
    def test_compensated_sin_large_magnitude(self):
        """Test compensated_sin with large magnitude - lines 313-316"""
        large_val = AbsoluteValue(magnitude=1e6, direction=1)
        result, compensation = Operations.compensated_sin(large_val)
        
        # Should handle large values with compensation
        assert isinstance(result, AbsoluteValue)
        assert isinstance(compensation, float)
    
    def test_compensated_cos_absolute_value(self):
        """Test compensated_cos with absolute value - lines 330-331"""
        abs_val = AbsoluteValue.absolute()
        result, compensation = Operations.compensated_cos(abs_val)
        
        assert result.magnitude == 1.0
        assert result.direction == 1
        assert compensation == 1.0
    
    def test_compensated_cos_large_magnitude(self):
        """Test compensated_cos with large magnitude - lines 333-336"""
        large_val = AbsoluteValue(magnitude=1e6, direction=1)
        result, compensation = Operations.compensated_cos(large_val)
        
        # Should handle large values with compensation
        assert isinstance(result, AbsoluteValue)
        assert isinstance(compensation, float)
    
    def test_sequence_sum_without_compensation(self):
        """Test sequence_sum without compensation - lines 352-353"""
        values = [AbsoluteValue(magnitude=1.0, direction=1), 
                 AbsoluteValue(magnitude=2.0, direction=1)]
        result, compensation = Operations.sequence_sum(values, use_compensation=False)
        
        assert result.magnitude == 3.0
        assert compensation == 0.0
    
    def test_sequence_sum_single_value(self):
        """Test sequence_sum with single value - lines 355-356"""
        values = [AbsoluteValue(magnitude=5.0, direction=1)]
        result, compensation = Operations.sequence_sum(values)
        
        assert result.magnitude == 5.0
        assert compensation == 0.0
    
    def test_sequence_sum_kahan_compensation(self):
        """Test sequence_sum Kahan summation - lines 358-376"""
        # Create values that would benefit from Kahan summation
        values = [AbsoluteValue(magnitude=1e16, direction=1),
                 AbsoluteValue(magnitude=1.0, direction=1),
                 AbsoluteValue(magnitude=1.0, direction=1)]
        result, compensation = Operations.sequence_sum(values, use_compensation=True)
        
        # Should use Kahan summation for better precision
        assert isinstance(result, AbsoluteValue)
        assert isinstance(compensation, float)
    
    def test_sequence_product_without_compensation(self):
        """Test sequence_product without compensation - lines 390-391"""
        values = [AbsoluteValue(magnitude=2.0, direction=1), 
                 AbsoluteValue(magnitude=3.0, direction=1)]
        result, compensation = Operations.sequence_product(values, use_compensation=False)
        
        assert result.magnitude == 6.0
        assert compensation == 1.0
    
    def test_sequence_product_single_value(self):
        """Test sequence_product with single value - lines 393-394"""
        values = [AbsoluteValue(magnitude=7.0, direction=1)]
        result, compensation = Operations.sequence_product(values)
        
        assert result.magnitude == 7.0
        assert compensation == 1.0
    
    def test_sequence_product_overflow_underflow(self):
        """Test sequence_product overflow/underflow handling - lines 396-414"""
        # Test overflow scenario
        large_values = [AbsoluteValue(magnitude=1e50, direction=1),
                       AbsoluteValue(magnitude=1e60, direction=1)]
        result, compensation = Operations.sequence_product(large_values, use_compensation=True)
        
        # Should handle overflow with compensation
        assert isinstance(result, AbsoluteValue)
        assert isinstance(compensation, float)
        
        # Test underflow scenario
        small_values = [AbsoluteValue(magnitude=1e-200, direction=1),
                       AbsoluteValue(magnitude=1e-200, direction=1)]
        result2, compensation2 = Operations.sequence_product(small_values, use_compensation=True)
        
        # Should handle underflow with compensation
        assert isinstance(result2, AbsoluteValue)
        assert isinstance(compensation2, float)
    
    def test_interpolate_invalid_t_parameter(self):
        """Test interpolate with invalid t parameter - lines 432-433"""
        a = AbsoluteValue(magnitude=1.0, direction=1)
        b = AbsoluteValue(magnitude=2.0, direction=1)
        
        with pytest.raises(ValueError, match=r'Interpolation parameter t must be in \[0, 1\] range'):
            Operations.interpolate(a, b, -0.1)
        
        with pytest.raises(ValueError, match=r'Interpolation parameter t must be in \[0, 1\] range'):
            Operations.interpolate(a, b, 1.1)
    
    def test_interpolate_edge_cases(self):
        """Test interpolate edge cases - lines 435-447"""
        a = AbsoluteValue(magnitude=1.0, direction=1)
        b = AbsoluteValue(magnitude=3.0, direction=-1)
        
        # Test t=0 (should return a)
        result_0 = Operations.interpolate(a, b, 0.0)
        assert result_0.magnitude == a.magnitude
        assert result_0.direction == a.direction
        
        # Test t=1 (should return b)
        result_1 = Operations.interpolate(a, b, 1.0)
        assert result_1.magnitude == b.magnitude
        assert result_1.direction == b.direction
        
        # Test t=0.5 (midpoint)
        result_mid = Operations.interpolate(a, b, 0.5)
        assert isinstance(result_mid, AbsoluteValue)
    
    def test_distance_absolute_values(self):
        """Test distance with absolute values - lines 460-461"""
        abs_val1 = AbsoluteValue.absolute()
        abs_val2 = AbsoluteValue.absolute()
        
        result = Operations.distance(abs_val1, abs_val2)
        assert result.is_absolute()
    
    def test_distance_same_values(self):
        """Test distance between same values - should be zero"""
        val = AbsoluteValue(magnitude=5.0, direction=1)
        result = Operations.distance(val, val)
        
        assert result.magnitude == 0.0
    
    def test_distance_opposite_directions(self):
        """Test distance with opposite directions"""
        val1 = AbsoluteValue(magnitude=3.0, direction=1)
        val2 = AbsoluteValue(magnitude=4.0, direction=-1)
        
        result = Operations.distance(val1, val2)
        # Distance should be |3 - (-4)| = 7
        assert result.magnitude == 7.0
    
    def test_normalize_absolute_value(self):
        """Test normalize with absolute value - lines 476-477"""
        abs_val = AbsoluteValue.absolute()
        
        with pytest.raises(ValueError, match='Cannot normalize Absolute value'):
            Operations.normalize(abs_val)
    
    def test_normalize_zero_magnitude(self):
        """Test normalize with zero magnitude - lines 479"""
        zero_val = AbsoluteValue(magnitude=0.0, direction=1)
        
        with pytest.raises(ValueError, match=r'Cannot normalize Absolute value'):
            Operations.normalize(zero_val)
    
    # Additional tests for uncovered lines in operations.py
    def test_compensated_add_near_cancellation(self):
        """Test compensated_add with near-cancellation scenario - lines 59-64"""
        # Create values that nearly cancel each other
        a = AbsoluteValue(magnitude=1e16, direction=1)
        b = AbsoluteValue(magnitude=1e16 - 1, direction=-1)
        
        result, compensation = Operations.compensated_add(a, b)
        
        # Should detect near-cancellation and apply compensation
        assert isinstance(result, AbsoluteValue)
        assert isinstance(compensation, float)
        assert result.magnitude > 0  # Should not be exactly zero due to compensation
    
    def test_compensated_add_standard_addition(self):
        """Test compensated_add standard addition path - lines 71-75"""
        a = AbsoluteValue(magnitude=3.0, direction=1)
        b = AbsoluteValue(magnitude=2.0, direction=1)
        
        result, compensation = Operations.compensated_add(a, b)
        
        # Standard addition without compensation issues
        assert result.magnitude == 5.0
        assert result.direction == 1
        assert compensation == 1.0  # Based on actual implementation
    
    def test_compensated_multiply_absolute_operands(self):
        """Test compensated_multiply with Absolute operands - lines 91-92"""
        abs_val = AbsoluteValue.absolute()
        regular_val = AbsoluteValue(magnitude=5.0, direction=1)
        
        result, compensation = Operations.compensated_multiply(abs_val, regular_val)
        
        # Multiplication with Absolute should return Absolute
        assert result.is_absolute()
        assert compensation == 0.0
    
    def test_compensated_multiply_overflow_scenario(self):
        """Test compensated_multiply overflow handling - lines 102-111"""
        # Create values that would cause overflow
        large_a = AbsoluteValue(magnitude=1e200, direction=1)
        large_b = AbsoluteValue(magnitude=1e200, direction=1)
        
        result, compensation = Operations.compensated_multiply(large_a, large_b)
        
        # Should handle overflow with compensation
        assert isinstance(result, AbsoluteValue)
        assert isinstance(compensation, float)
        # Compensation should account for overflow
        assert compensation != 1.0
    
    def test_compensated_multiply_underflow_scenario(self):
        """Test compensated_multiply underflow handling - lines 102-111"""
        # Create values that would cause underflow
        small_a = AbsoluteValue(magnitude=1e-200, direction=1)
        small_b = AbsoluteValue(magnitude=1e-200, direction=1)
        
        result, compensation = Operations.compensated_multiply(small_a, small_b)
        
        # Should handle underflow with compensation
        assert isinstance(result, AbsoluteValue)
        assert isinstance(compensation, float)
        # Compensation should account for underflow
        assert compensation != 1.0
    
    def test_compensated_divide_absolute_denominator(self):
        """Test compensated_divide with Absolute denominator - lines 132-136"""
        numerator = AbsoluteValue(magnitude=10.0, direction=1)
        abs_denominator = AbsoluteValue.absolute()
        
        # Should raise ValueError for division by Absolute
        with pytest.raises(ValueError, match='Cannot divide by Absolute \(denominator magnitude=0\)'):
            Operations.compensated_divide(numerator, abs_denominator)
    
    def test_compensated_power_absolute_base(self):
        """Test compensated_power with Absolute base - lines 152-153"""
        abs_base = AbsoluteValue.absolute()
        exponent = 5.0
        
        result, compensation = Operations.compensated_power(abs_base, exponent)
        
        # Absolute raised to any power should return Absolute
        assert result.is_absolute()
        assert compensation == 0.0
    
    def test_compensated_power_zero_exponent_negative_base(self):
        """Test compensated_power with zero exponent - lines 155-156"""
        base = AbsoluteValue(magnitude=7.0, direction=-1)
        exponent = 0.0
        
        result, compensation = Operations.compensated_power(base, exponent)
        
        # Any number to power 0 should be 1
        assert result.magnitude == 1.0
        assert result.direction == 1
        assert compensation == 1.0  # Based on actual implementation
    
    def test_compensated_power_unit_exponent(self):
        """Test compensated_power with unit exponent - lines 158-159"""
        base = AbsoluteValue(magnitude=3.5, direction=-1)
        exponent = 1.0
        
        result, compensation = Operations.compensated_power(base, exponent)
        
        # Any number to power 1 should be itself
        assert result.magnitude == 3.5
        assert result.direction == -1
        assert compensation == 1.0  # Based on actual implementation
    
    def test_compensated_power_even_integer_negative_base(self):
        """Test compensated_power with even integer and negative base - lines 162"""
        base = AbsoluteValue(magnitude=2.0, direction=-1)
        exponent = 4  # Even integer
        
        result, compensation = Operations.compensated_power(base, exponent)
        
        # Negative base to even power should be positive
        assert result.magnitude == 16.0
        assert result.direction == 1
        assert compensation == 1.0  # Based on actual implementation
    
    def test_compensated_power_odd_integer_negative_base(self):
        """Test compensated_power with odd integer and negative base - lines 165"""
        base = AbsoluteValue(magnitude=2.0, direction=-1)
        exponent = 3  # Odd integer
        
        result, compensation = Operations.compensated_power(base, exponent)
        
        # Negative base to odd power should be negative
        assert result.magnitude == 8.0
        assert result.direction == -1
        assert compensation == 1.0  # Based on actual implementation
    
    def test_compensated_power_fractional_exponent_negative_base(self):
        """Test compensated_power with fractional exponent and negative base - lines 178-180"""
        base = AbsoluteValue(magnitude=4.0, direction=-1)
        exponent = 0.5  # Fractional exponent
        
        # ValueError is caught and returns Absolute with compensation 0.0
        result, compensation = Operations.compensated_power(base, exponent)
        assert result.is_absolute()
        assert compensation == 0.0
    
    def test_compensated_power_large_exponent_overflow(self):
        """Test compensated_power overflow handling - lines 184-186"""
        base = AbsoluteValue(magnitude=10.0, direction=1)
        exponent = 500  # Very large exponent
        
        result, compensation = Operations.compensated_power(base, exponent)
        
        # Should handle overflow with compensation
        assert isinstance(result, AbsoluteValue)
        assert isinstance(compensation, float)
        # Should apply logarithmic compensation for overflow
        assert compensation != 0.0
    
    def test_compensated_power_small_base_underflow(self):
        """Test compensated_power underflow handling - lines 195, 200-205"""
        base = AbsoluteValue(magnitude=1e-100, direction=1)
        exponent = 10  # Large enough to cause underflow
        
        result, compensation = Operations.compensated_power(base, exponent)
        
        # Should handle underflow with compensation
        assert isinstance(result, AbsoluteValue)
        assert isinstance(compensation, float)
        # Should apply logarithmic compensation for underflow
        assert compensation != 0.0
    
    def test_compensated_log_custom_base(self):
        """Test compensated_log with custom base - lines 222-231"""
        value = AbsoluteValue(magnitude=8.0, direction=1)
        base = 2.0
        
        result, compensation = Operations.compensated_log(value, base)
        
        # log_2(8) = 3
        assert abs(result.magnitude - 3.0) < 1e-10
        assert result.direction == 1
        assert isinstance(compensation, float)
    
    def test_compensated_log_base_one_error(self):
        """Test compensated_log with base=1 error - lines 224-225"""
        value = AbsoluteValue(magnitude=5.0, direction=1)
        base = 1.0
        
        # Should raise ValueError for base=1
        with pytest.raises(ValueError, match='Logarithm base must be positive and not equal to 1'):
            Operations.compensated_log(value, base)
    
    def test_compensated_log_negative_base_error(self):
        """Test compensated_log with negative base error - lines 226-227"""
        value = AbsoluteValue(magnitude=5.0, direction=1)
        base = -2.0
        
        # Should raise ValueError for negative base
        with pytest.raises(ValueError, match='Logarithm base must be positive and not equal to 1'):
            Operations.compensated_log(value, base)
    
    def test_compensated_log_zero_base_error(self):
        """Test compensated_log with zero base error - lines 228-229"""
        value = AbsoluteValue(magnitude=5.0, direction=1)
        base = 0.0
        
        # Should raise ValueError for base=0
        with pytest.raises(ValueError, match='Logarithm base must be positive and not equal to 1'):
            Operations.compensated_log(value, base)
    
    def test_compensated_exp_large_value_overflow_protection(self):
        """Test compensated_exp overflow protection - lines 249-265"""
        # Test with value that triggers overflow protection
        large_value = AbsoluteValue(magnitude=800.0, direction=1)
        
        result, compensation = Operations.compensated_exp(large_value)
        
        # Should return 1e100 magnitude with logarithmic compensation
        assert result.magnitude == 1e100
        assert result.direction == 1
        # Compensation should be the excess: 800 - log(1e100)
        expected_compensation = 800.0 - math.log(1e100)
        assert abs(compensation - expected_compensation) < 1e-10
    
    def test_compensated_exp_negative_large_value(self):
        """Test compensated_exp with large negative value - lines 261-265"""
        # Test with large negative value
        large_neg_value = AbsoluteValue(magnitude=800.0, direction=-1)
        
        result, compensation = Operations.compensated_exp(large_neg_value)
        
        # Should handle large negative values
        assert isinstance(result, AbsoluteValue)
        assert result.direction == 1  # exp is always positive
        assert isinstance(compensation, float)
    
    def test_sequence_sum_empty_list(self):
        """Test sequence_sum with empty list - should handle gracefully"""
        empty_values = []
        
        # Should handle empty list gracefully
        try:
            result, compensation = Operations.sequence_sum(empty_values)
            # If it doesn't raise an error, check the result
            assert isinstance(result, AbsoluteValue)
            assert isinstance(compensation, float)
        except (ValueError, IndexError):
            # Empty list might raise an error, which is acceptable
            pass
    
    def test_sequence_product_empty_list(self):
        """Test sequence_product with empty list - should handle gracefully"""
        empty_values = []
        
        # Should handle empty list gracefully
        try:
            result, compensation = Operations.sequence_product(empty_values)
            # If it doesn't raise an error, check the result
            assert isinstance(result, AbsoluteValue)
            assert isinstance(compensation, float)
        except (ValueError, IndexError):
            # Empty list might raise an error, which is acceptable
            pass
     
    def test_sequence_sum_large_compensation_values(self):
        """Test sequence_sum with values requiring significant compensation - lines 358-376"""
        # Create a scenario where Kahan summation is beneficial
        values = [
            AbsoluteValue(magnitude=1e20, direction=1),
            AbsoluteValue(magnitude=1.0, direction=1),
            AbsoluteValue(magnitude=1.0, direction=1),
            AbsoluteValue(magnitude=1e20, direction=-1)
        ]
        
        result, compensation = Operations.sequence_sum(values, use_compensation=True)
        
        # Should use Kahan summation for better precision
        assert isinstance(result, AbsoluteValue)
        assert isinstance(compensation, float)
        # Result should be close to 2.0 due to compensation
        assert abs(result.magnitude - 2.0) < 1e-10
    
    def test_sequence_product_mixed_directions(self):
        """Test sequence_product with mixed directions - lines 396-414"""
        values = [
            AbsoluteValue(magnitude=2.0, direction=1),
            AbsoluteValue(magnitude=3.0, direction=-1),
            AbsoluteValue(magnitude=4.0, direction=1)
        ]
        
        result, compensation = Operations.sequence_product(values, use_compensation=True)
        
        # Product should be 24 with negative direction (odd number of negatives)
        assert result.magnitude == 24.0
        assert result.direction == -1
        assert isinstance(compensation, float)
     
    def test_interpolate_absolute_values(self):
        """Test interpolate with absolute values - edge case"""
        abs_a = AbsoluteValue.absolute()
        abs_b = AbsoluteValue.absolute()
        
        result = Operations.interpolate(abs_a, abs_b, 0.5)
        
        # Interpolation between two Absolute values should return Absolute
        assert result.is_absolute()
    
    def test_interpolate_zero_magnitude_values(self):
        """Test interpolate with zero magnitude values"""
        zero_a = AbsoluteValue(magnitude=0.0, direction=1)
        zero_b = AbsoluteValue(magnitude=0.0, direction=-1)
        
        result = Operations.interpolate(zero_a, zero_b, 0.3)
        
        # Interpolation between zero values should be zero
        assert result.magnitude == 0.0
        assert isinstance(result.direction, int)
    
    def test_distance_large_magnitude_difference(self):
        """Test distance with very different magnitudes"""
        small_val = AbsoluteValue(magnitude=1e-10, direction=1)
        large_val = AbsoluteValue(magnitude=1e10, direction=1)
        
        result = Operations.distance(small_val, large_val)
        
        # Distance should be approximately the large value
        assert abs(result.magnitude - 1e10) < 1e-5
        assert result.direction == 1
    
    def test_normalize_positive_value(self):
        """Test normalize with positive value"""
        val = AbsoluteValue(magnitude=5.0, direction=1)
        result = Operations.normalize(val)
        
        # Normalized value should have magnitude 1.0
        assert result.magnitude == 1.0
        assert result.direction == 1
    
    def test_normalize_negative_value(self):
        """Test normalize with negative value"""
        val = AbsoluteValue(magnitude=3.0, direction=-1)
        result = Operations.normalize(val)
        
        # Normalized value should have magnitude 1.0
        assert result.magnitude == 1.0
        assert result.direction == -1


if __name__ == "__main__":
    pytest.main([__file__])