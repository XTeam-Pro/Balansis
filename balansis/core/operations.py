"""Core operations module for Balansis library.

This module implements compensated arithmetic operations that maintain mathematical
stability and avoid traditional issues with zero division and infinity. All operations
are designed around the Absolute Compensation Theory (ACT) principles.
"""

import math
from typing import Union, List, Tuple, Optional
from decimal import Decimal, getcontext

from .absolute import AbsoluteValue
from .eternity import EternalRatio

# Set high precision for decimal operations
getcontext().prec = 50

# Type aliases for clarity
NumericType = Union[float, int, AbsoluteValue, EternalRatio]
CompensatedResult = Tuple[AbsoluteValue, float]  # (result, compensation_factor)
CompensatedDivideResult = Tuple[EternalRatio, float]  # (ratio, compensation_factor)


class Operations:
    """Core operations implementing Absolute Compensation Theory.
    
    This class provides static methods for performing compensated arithmetic
    operations that maintain stability and avoid mathematical singularities.
    All operations follow ACT principles of compensation, stability, and eternity.
    """
    
    # Mathematical constants for ACT
    COMPENSATION_THRESHOLD = 1e-15
    STABILITY_FACTOR = 1e-12
    MAX_COMPENSATION_ITERATIONS = 100
    
    @staticmethod
    def compensated_add(a: AbsoluteValue, b: AbsoluteValue, 
                       compensation_factor: float = 1.0) -> CompensatedResult:
        """Perform compensated addition of two AbsoluteValues.
        
        This operation ensures mathematical stability by applying compensation
        when values approach critical thresholds that could cause instability.
        
        Args:
            a: First AbsoluteValue operand
            b: Second AbsoluteValue operand
            compensation_factor: Factor to adjust compensation strength
            
        Returns:
            Tuple of (result_AbsoluteValue, applied_compensation_factor)
            
        Examples:
            >>> a = AbsoluteValue(magnitude=1e-16, direction=1)
            >>> b = AbsoluteValue(magnitude=1e-16, direction=-1)
            >>> result, comp = Operations.compensated_add(a, b)
            >>> result.is_absolute()  # True - compensated to Absolute
        """
        # Check if operands are near-canceling
        if (abs(a.magnitude - b.magnitude) < Operations.COMPENSATION_THRESHOLD and 
            a.direction != b.direction):
            # Apply compensation to prevent numerical instability
            compensation = compensation_factor * Operations.STABILITY_FACTOR
            result = AbsoluteValue.absolute()  # Result is Absolute
            return result, compensation
        
        # Standard addition with stability check
        result = a + b
        
        # Check if result needs compensation
        if result.magnitude < Operations.COMPENSATION_THRESHOLD:
            compensated_result = AbsoluteValue.absolute()
            applied_compensation = result.magnitude / Operations.COMPENSATION_THRESHOLD
            return compensated_result, applied_compensation
        
        return result, 1.0
    
    @staticmethod
    def compensated_multiply(a: AbsoluteValue, b: AbsoluteValue,
                           compensation_factor: float = 1.0) -> CompensatedResult:
        """Perform compensated multiplication of two AbsoluteValues.
        
        Args:
            a: First AbsoluteValue operand
            b: Second AbsoluteValue operand
            compensation_factor: Factor to adjust compensation strength
            
        Returns:
            Tuple of (result_AbsoluteValue, applied_compensation_factor)
        """
        # Handle Absolute operands
        if a.is_absolute() or b.is_absolute():
            return AbsoluteValue.absolute(), 0.0
        
        # Standard multiplication
        result_magnitude = a.magnitude * b.magnitude
        result_direction = a.direction * b.direction
        
        # Check for overflow compensation
        if result_magnitude > 1e100:
            # Apply logarithmic compensation
            log_compensation = math.log10(result_magnitude) - 100
            compensated_magnitude = 1e100
            result = AbsoluteValue(magnitude=compensated_magnitude, direction=result_direction)
            return result, log_compensation
        
        # Check for underflow compensation
        if result_magnitude < Operations.COMPENSATION_THRESHOLD:
            return AbsoluteValue.absolute(), result_magnitude
        
        result = AbsoluteValue(magnitude=result_magnitude, direction=result_direction)
        return result, 1.0
    
    @staticmethod
    def compensated_divide(numerator: AbsoluteValue, denominator: AbsoluteValue,
                          compensation_factor: float = 1.0) -> "CompensatedDivideResult":
        """Perform compensated division using EternalRatio.

        This operation avoids traditional division issues by creating a structural
        ratio that maintains mathematical stability.

        Args:
            numerator: AbsoluteValue to divide.
            denominator: AbsoluteValue to divide by.
            compensation_factor: Factor to adjust compensation strength.

        Returns:
            Tuple of (EternalRatio result, applied_compensation_factor).

        Raises:
            ValueError: If denominator is Absolute (magnitude=0).
        """
        if denominator.is_absolute():
            raise ValueError('Cannot divide by Absolute (denominator magnitude=0)')

        # Detect near-Absolute denominator and compute compensation
        applied_compensation = 1.0
        if denominator.magnitude < Operations.COMPENSATION_THRESHOLD:
            applied_compensation = denominator.magnitude / Operations.COMPENSATION_THRESHOLD

        # Create EternalRatio for stable division
        ratio = EternalRatio(numerator=numerator, denominator=denominator)
        return ratio, applied_compensation
    
    @staticmethod
    def compensated_power(base: AbsoluteValue, exponent: float,
                         compensation_factor: float = 1.0) -> CompensatedResult:
        """Perform compensated exponentiation.
        
        Args:
            base: AbsoluteValue base
            exponent: Power to raise base to
            compensation_factor: Factor to adjust compensation strength
            
        Returns:
            Tuple of (result_AbsoluteValue, applied_compensation_factor)
        """
        # Handle special cases
        if base.is_absolute():
            if exponent > 0:
                return AbsoluteValue.absolute(), 0.0
            elif exponent == 0:
                return AbsoluteValue.unit_positive(), 1.0
            else:
                # Negative exponent with Absolute base is undefined
                raise ValueError('Cannot raise Absolute to negative power')
        
        if exponent == 0:
            return AbsoluteValue.unit_positive(), 1.0
        
        if exponent == 1:
            return base, 1.0
        
        # Calculate power with overflow/underflow protection
        try:
            result_magnitude = base.magnitude ** abs(exponent)
            
            # Handle direction for fractional exponents
            if exponent == int(exponent):  # Integer exponent
                if int(exponent) % 2 == 0:
                    result_direction = 1  # Even power is always positive
                else:
                    result_direction = base.direction if exponent > 0 else base.direction
            else:  # Fractional exponent
                if base.direction < 0:
                    raise ValueError('Cannot raise negative AbsoluteValue to fractional power')
                result_direction = 1
            
            # Apply negative exponent
            if exponent < 0:
                if result_magnitude == 0:
                    raise ValueError('Cannot invert zero magnitude')
                result_magnitude = 1.0 / result_magnitude
            
            # Check for overflow/underflow
            if result_magnitude > 1e100:
                log_compensation = math.log10(result_magnitude) - 100
                result = AbsoluteValue(magnitude=1e100, direction=result_direction)
                return result, log_compensation
            
            if result_magnitude < Operations.COMPENSATION_THRESHOLD:
                return AbsoluteValue.absolute(), result_magnitude
            
            result = AbsoluteValue(magnitude=result_magnitude, direction=result_direction)
            return result, 1.0
            
        except (OverflowError, ValueError) as e:
            # Apply compensation for mathematical errors
            if 'overflow' in str(e).lower():
                return AbsoluteValue(magnitude=1e100, direction=1), float('inf')
            else:
                return AbsoluteValue.absolute(), 0.0
    
    @staticmethod
    def compensated_sqrt(value: AbsoluteValue, 
                        compensation_factor: float = 1.0) -> CompensatedResult:
        """Perform compensated square root operation.
        
        Args:
            value: AbsoluteValue to take square root of
            compensation_factor: Factor to adjust compensation strength
            
        Returns:
            Tuple of (result_AbsoluteValue, applied_compensation_factor)
            
        Raises:
            ValueError: If value has negative direction
        """
        if value.direction < 0:
            raise ValueError('Cannot take square root of negative AbsoluteValue')
        
        if value.is_absolute():
            return AbsoluteValue.absolute(), 0.0
        
        result_magnitude = math.sqrt(value.magnitude)
        result = AbsoluteValue(magnitude=result_magnitude, direction=1)
        
        return result, 1.0
    
    @staticmethod
    def compensated_log(value: AbsoluteValue, base: float = math.e,
                       compensation_factor: float = 1.0) -> CompensatedResult:
        """Perform compensated logarithm operation.
        
        Args:
            value: AbsoluteValue to take logarithm of
            base: Logarithm base (default: natural log)
            compensation_factor: Factor to adjust compensation strength
            
        Returns:
            Tuple of (result_AbsoluteValue, applied_compensation_factor)
            
        Raises:
            ValueError: If value is Absolute or has negative direction
        """
        if value.is_absolute():
            raise ValueError('Cannot take logarithm of Absolute')
        
        if value.direction < 0:
            raise ValueError('Cannot take logarithm of negative AbsoluteValue')
        
        if base <= 0 or base == 1:
            raise ValueError('Logarithm base must be positive and not equal to 1')
        
        # Calculate logarithm
        if base == math.e:
            log_value = math.log(value.magnitude)
        else:
            log_value = math.log(value.magnitude) / math.log(base)
        
        result = AbsoluteValue.from_float(log_value)
        return result, 1.0
    
    @staticmethod
    def compensated_exp(value: AbsoluteValue,
                       compensation_factor: float = 1.0) -> CompensatedResult:
        """Perform compensated exponential operation.
        
        Args:
            value: AbsoluteValue exponent
            compensation_factor: Factor to adjust compensation strength
            
        Returns:
            Tuple of (result_AbsoluteValue, applied_compensation_factor)
        """
        if value.is_absolute():
            return AbsoluteValue.unit_positive(), 1.0
        
        # Calculate exponential with overflow protection
        try:
            exp_value = math.exp(value.to_float())
            
            if exp_value > 1e100:
                # Apply logarithmic compensation
                log_compensation = value.to_float() - 100 * math.log(10)
                result = AbsoluteValue(magnitude=1e100, direction=1)
                return result, log_compensation
            
            result = AbsoluteValue(magnitude=exp_value, direction=1)
            return result, 1.0
            
        except OverflowError:
            return AbsoluteValue(magnitude=1e100, direction=1), float('inf')
    
    @staticmethod
    def compensated_sin(value: AbsoluteValue,
                       compensation_factor: float = 1.0) -> CompensatedResult:
        """Perform compensated sine operation.
        
        Args:
            value: AbsoluteValue angle (in radians)
            compensation_factor: Factor to adjust compensation strength
            
        Returns:
            Tuple of (result_AbsoluteValue, applied_compensation_factor)
        """
        if value.is_absolute():
            return AbsoluteValue.absolute(), 0.0
        
        sin_value = math.sin(value.to_float())
        result = AbsoluteValue.from_float(sin_value)
        
        return result, 1.0
    
    @staticmethod
    def compensated_cos(value: AbsoluteValue,
                       compensation_factor: float = 1.0) -> CompensatedResult:
        """Perform compensated cosine operation.
        
        Args:
            value: AbsoluteValue angle (in radians)
            compensation_factor: Factor to adjust compensation strength
            
        Returns:
            Tuple of (result_AbsoluteValue, applied_compensation_factor)
        """
        if value.is_absolute():
            return AbsoluteValue.unit_positive(), 1.0
        
        cos_value = math.cos(value.to_float())
        result = AbsoluteValue.from_float(cos_value)
        
        return result, 1.0
    
    @staticmethod
    def sequence_sum(values: List[AbsoluteValue],
                    use_compensation: bool = True) -> CompensatedResult:
        """Calculate compensated sum of a sequence of AbsoluteValues.
        
        Uses Kahan summation algorithm for improved numerical stability.
        
        Args:
            values: List of AbsoluteValues to sum
            use_compensation: Whether to apply Kahan compensation
            
        Returns:
            Tuple of (sum_AbsoluteValue, total_compensation_error)
        """
        if not values:
            return AbsoluteValue.absolute(), 0.0
        
        if len(values) == 1:
            return values[0], 0.0
        
        if not use_compensation:
            # Simple summation
            result = values[0]
            for value in values[1:]:
                result = result + value
            return result, 0.0
        
        # Kahan summation for improved accuracy
        total = values[0]
        compensation_error = AbsoluteValue.absolute()
        
        for value in values[1:]:
            # Compensated addition
            compensated_value = value - compensation_error
            new_total = total + compensated_value
            compensation_error = (new_total - total) - compensated_value
            total = new_total
        
        return total, compensation_error.magnitude
    
    @staticmethod
    def sequence_product(values: List[AbsoluteValue],
                        use_compensation: bool = True) -> CompensatedResult:
        """Calculate compensated product of a sequence of AbsoluteValues.
        
        Args:
            values: List of AbsoluteValues to multiply
            use_compensation: Whether to apply overflow/underflow compensation
            
        Returns:
            Tuple of (product_AbsoluteValue, total_compensation_factor)
        """
        if not values:
            return AbsoluteValue.unit_positive(), 1.0
        
        if len(values) == 1:
            return values[0], 1.0
        
        # Check for any Absolute values
        if any(v.is_absolute() for v in values):
            return AbsoluteValue.absolute(), 0.0
        
        # Calculate product with compensation
        result = values[0]
        total_compensation = 1.0
        
        for value in values[1:]:
            if use_compensation:
                result, comp_factor = Operations.compensated_multiply(result, value)
                total_compensation *= comp_factor
            else:
                result = AbsoluteValue(
                    magnitude=result.magnitude * value.magnitude,
                    direction=result.direction * value.direction
                )
        
        return result, total_compensation
    
    @staticmethod
    def interpolate(start: AbsoluteValue, end: AbsoluteValue, 
                   t: float) -> AbsoluteValue:
        """Perform linear interpolation between two AbsoluteValues.
        
        Args:
            start: Starting AbsoluteValue
            end: Ending AbsoluteValue
            t: Interpolation parameter (0.0 to 1.0)
            
        Returns:
            Interpolated AbsoluteValue
            
        Raises:
            ValueError: If t is not in [0, 1] range
        """
        if not (0.0 <= t <= 1.0):
            raise ValueError('Interpolation parameter t must be in [0, 1] range')
        
        if t == 0.0:
            return start
        if t == 1.0:
            return end
        
        # Linear interpolation: start + t * (end - start)
        difference = end - start
        scaled_diff = AbsoluteValue(
            magnitude=difference.magnitude * t,
            direction=difference.direction
        )
        
        return start + scaled_diff
    
    @staticmethod
    def distance(a: AbsoluteValue, b: AbsoluteValue) -> AbsoluteValue:
        """Calculate the distance between two AbsoluteValues.
        
        Args:
            a: First AbsoluteValue
            b: Second AbsoluteValue
            
        Returns:
            AbsoluteValue representing the distance (always positive)
        """
        difference = a - b
        return AbsoluteValue(magnitude=difference.magnitude, direction=1)
    
    @staticmethod
    def normalize(value: AbsoluteValue) -> AbsoluteValue:
        """Normalize an AbsoluteValue to unit magnitude.
        
        Args:
            value: AbsoluteValue to normalize
            
        Returns:
            AbsoluteValue with magnitude 1.0 and same direction
            
        Raises:
            ValueError: If value is Absolute (cannot normalize)
        """
        if value.is_absolute():
            raise ValueError('Cannot normalize Absolute value')
        
        return AbsoluteValue(magnitude=1.0, direction=value.direction)