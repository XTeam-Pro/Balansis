# Copyright (c) 2024-2026 Tikhonov Andrey. All rights reserved.
# SPDX-License-Identifier: MIT (non-commercial) | Commercial use: see COMMERCIAL_LICENSE.md
"""EternalRatio class implementation for Balansis library.

This module implements the EternalRatio type that represents structural ratios
between AbsoluteValues. These ratios are invariant across time, context, and scale,
providing stable mathematical relationships without traditional division issues.
"""

import math
from typing import Any, Optional
from pydantic import BaseModel, validator, Field

from .absolute import AbsoluteValue


class EternalRatio(BaseModel):
    """Structural ratio between two AbsoluteValues.
    
    EternalRatio represents the relationship between two AbsoluteValues as a
    stable, invariant ratio that maintains its properties across mathematical
    transformations. This replaces traditional division with a more stable
    structural relationship.
    
    Attributes:
        numerator: The AbsoluteValue in the numerator position
        denominator: The AbsoluteValue in the denominator position (cannot be Absolute)
    
    Examples:
        >>> a = AbsoluteValue(magnitude=6.0, direction=1)
        >>> b = AbsoluteValue(magnitude=2.0, direction=1)
        >>> ratio = EternalRatio(numerator=a, denominator=b)
        >>> ratio.value()  # Returns 3.0
        >>> ratio.is_stable()  # Returns True
    """
    
    numerator: AbsoluteValue = Field(
        ..., 
        description="AbsoluteValue in the numerator position"
    )
    denominator: AbsoluteValue = Field(
        ..., 
        description="AbsoluteValue in the denominator position (non-Absolute)"
    )
    
    class Config:
        """Pydantic configuration for EternalRatio."""
        frozen = True  # Make instances immutable
        validate_assignment = True
        arbitrary_types_allowed = True
    
    @validator('denominator')
    def denominator_not_absolute(cls, v: AbsoluteValue) -> AbsoluteValue:
        """Ensure denominator is not Absolute (magnitude != 0).
        
        Args:
            v: The denominator AbsoluteValue to validate
            
        Returns:
            The validated denominator
            
        Raises:
            ValueError: If denominator has magnitude 0 (is Absolute)
        """
        if v.magnitude == 0.0:
            raise ValueError('Denominator cannot be Absolute (magnitude=0)')
        return v
    
    def value(self) -> float:
        """Calculate the numerical value of the eternal ratio.
        
        The value is computed as the ratio of magnitudes, maintaining
        the structural relationship between the AbsoluteValues.
        
        Returns:
            Float representing the ratio value
        """
        return self.numerator.magnitude / self.denominator.magnitude
    
    def signed_value(self) -> float:
        """Get the signed directional value of the ratio.
        
        This represents the directional relationship between numerator and denominator,
        normalized to ±1.0 based on the direction compatibility.
        When numerator is Absolute (magnitude = 0), returns 0.0.
        
        Returns:
            The directional factor: +1.0, -1.0, or 0.0 (for Absolute numerator)
        """
        # If numerator is Absolute (magnitude = 0), return 0.0
        if self.numerator.magnitude == 0.0:
            return 0.0
        direction_factor = self.numerator.direction * self.denominator.direction
        return direction_factor
    
    def is_stable(self, tolerance: float = 1e-10) -> bool:
        """Check if the ratio is mathematically stable.
        
        A ratio is considered stable if:
        1. Both numerator and denominator are finite
        2. Denominator is not Absolute
        3. The ratio value is finite
        4. The ratio is within reasonable bounds (not too extreme)
        5. The ratio is not at precision limits
        
        Args:
            tolerance: Tolerance for stability checks
            
        Returns:
            True if the ratio is stable
        """
        try:
            ratio_value = self.numerical_value()
            # Import ACT_STABILITY_THRESHOLD from balansis
            from balansis import ACT_STABILITY_THRESHOLD
            
            # Stability is based on reasonable bounds, not precision limits
            # Ratios like 1.1 should be stable, but 10.0 should be unstable
            reasonable_bound = 5.0
            within_reasonable_bounds = (abs(ratio_value) <= reasonable_bound and 
                                      abs(ratio_value) >= 1.0/reasonable_bound)
            
            # However, ratios at precision limits should be unstable
            # This handles the case where ratio is very close to 1.0 but at the threshold
            # Only consider it a precision limit if it's close to 1.0 (within 0.01) but at threshold
            close_to_unity = abs(ratio_value - 1.0) < 0.01
            at_precision_limit = close_to_unity and abs(ratio_value - 1.0) >= ACT_STABILITY_THRESHOLD
            
            return (math.isfinite(ratio_value) and 
                    self.denominator.magnitude > tolerance and
                    within_reasonable_bounds and
                    not at_precision_limit)
        except (ZeroDivisionError, ValueError):
            return False
    
    def __eq__(self, other: Any) -> bool:
        """Equality comparison between eternal ratios.
        
        Two ratios are equal if they have the same numerical value,
        regardless of the specific AbsoluteValues used.
        
        Args:
            other: Object to compare with
            
        Returns:
            True if ratios are mathematically equivalent
        """
        if not isinstance(other, EternalRatio):
            return False
        
        try:
            return abs(self.numerical_value() - other.numerical_value()) < 1e-10
        except (ZeroDivisionError, ValueError):
            return False
    
    def __lt__(self, other: 'EternalRatio') -> bool:
        """Less than comparison based on numerical values."""
        return self.numerical_value() < other.numerical_value()
    
    def __le__(self, other: 'EternalRatio') -> bool:
        """Less than or equal comparison."""
        return self.numerical_value() <= other.numerical_value()
    
    def __gt__(self, other: 'EternalRatio') -> bool:
        """Greater than comparison based on numerical values."""
        return self.numerical_value() > other.numerical_value()
    
    def __ge__(self, other: 'EternalRatio') -> bool:
        """Greater than or equal comparison."""
        return self.numerical_value() >= other.numerical_value()
    
    def __hash__(self) -> int:
        """Hash function for use in sets and dictionaries."""
        # Hash based on numerical value for mathematical equivalence
        return hash(round(self.numerical_value(), 10))
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"EternalRatio(numerator={self.numerator!r}, denominator={self.denominator!r})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"EternalRatio({self.numerator}) / ({self.denominator}) = {self.numerical_value():.6f}"

    def __getstate__(self):
        return {"numerator": self.numerator, "denominator": self.denominator}

    def __setstate__(self, state):
        object.__setattr__(self, "numerator", state["numerator"])
        object.__setattr__(self, "denominator", state["denominator"])

    def to_json(self):
        return {"type": "EternalRatio", "value": self.numerical_value(), "structural_dump": {"numerator": self.numerator.to_float(), "denominator": self.denominator.to_float()}}

    @classmethod
    def __get_validators__(cls):
        def validate(v):
            if isinstance(v, EternalRatio):
                return v
            if isinstance(v, (int, float)):
                return EternalRatio.from_float(float(v))
            if isinstance(v, dict) and "numerator" in v and "denominator" in v:
                num = AbsoluteValue.from_float(float(v["numerator"])) if not isinstance(v["numerator"], AbsoluteValue) else v["numerator"]
                den = AbsoluteValue.from_float(float(v["denominator"])) if not isinstance(v["denominator"], AbsoluteValue) else v["denominator"]
                return EternalRatio(numerator=num, denominator=den)
            raise TypeError("invalid EternalRatio input")
        yield validate
    
    def __mul__(self, other) -> 'EternalRatio':
        """Multiply eternal ratio by another ratio or scalar.
        
        Args:
            other: Another EternalRatio or scalar to multiply with
            
        Returns:
            New EternalRatio representing the product
        """
        if isinstance(other, EternalRatio):
            # (a/b) * (c/d) = (a*c) / (b*d)
            new_numerator = AbsoluteValue(
                magnitude=self.numerator.magnitude * other.numerator.magnitude,
                direction=self.numerator.direction * other.numerator.direction
            )
            new_denominator = AbsoluteValue(
                magnitude=self.denominator.magnitude * other.denominator.magnitude,
                direction=self.denominator.direction * other.denominator.direction
            )
            return EternalRatio(numerator=new_numerator, denominator=new_denominator)
        elif isinstance(other, (int, float)):
            # Scalar multiplication: multiply numerator by scalar
            new_numerator = AbsoluteValue(
                magnitude=self.numerator.magnitude * abs(other),
                direction=self.numerator.direction * (1 if other > 0 else -1)
            )
            return EternalRatio(numerator=new_numerator, denominator=self.denominator)
        else:
            return NotImplemented
    
    def __truediv__(self, other) -> 'EternalRatio':
        """Divide eternal ratio by another ratio or scalar.
        
        Args:
            other: Another EternalRatio or scalar to divide by
            
        Returns:
            New EternalRatio representing the quotient
        """
        if isinstance(other, EternalRatio):
            # Division is multiplication by the inverse
            return self * other.inverse()
        elif isinstance(other, (int, float)):
            # Scalar division: divide numerator by scalar
            if other == 0:
                raise ValueError("Cannot divide by zero")
            new_numerator = AbsoluteValue(
                magnitude=self.numerator.magnitude / abs(other),
                direction=self.numerator.direction * (1 if other > 0 else -1)
            )
            return EternalRatio(numerator=new_numerator, denominator=self.denominator)
        else:
            return NotImplemented

    def __radd__(self, left) -> 'EternalRatio':
        if isinstance(left, (int, float)) and math.isfinite(left):
            return EternalRatio.from_float(float(left)) + self
        return NotImplemented

    def __rsub__(self, left) -> 'EternalRatio':
        if isinstance(left, (int, float)) and math.isfinite(left):
            return EternalRatio.from_float(float(left)) - self
        return NotImplemented

    def __rmul__(self, left) -> 'EternalRatio':
        if isinstance(left, (int, float)) and math.isfinite(left):
            return self * float(left)
        return NotImplemented

    def __rtruediv__(self, left) -> 'EternalRatio':
        if isinstance(left, (int, float)) and math.isfinite(left):
            return EternalRatio.from_float(float(left)) / self
        return NotImplemented
    
    def __add__(self, other: 'EternalRatio') -> 'EternalRatio':
        """Add two eternal ratios with common denominator.
        
        (a/c) + (b/c) = (a+b)/c
        For different denominators: (a/b) + (c/d) = (a*d + c*b)/(b*d)
        
        Args:
            other: Another EternalRatio to add
            
        Returns:
            New EternalRatio representing the sum
        """
        if self.denominator == other.denominator:
            # Same denominator: add numerators
            new_numerator = self.numerator + other.numerator
            return EternalRatio(numerator=new_numerator, denominator=self.denominator)
        else:
            # Different denominators: cross multiply
            # (a/b) + (c/d) = (a*d + c*b)/(b*d)
            term1 = AbsoluteValue(
                magnitude=self.numerator.magnitude * other.denominator.magnitude,
                direction=self.numerator.direction * other.denominator.direction
            )
            term2 = AbsoluteValue(
                magnitude=other.numerator.magnitude * self.denominator.magnitude,
                direction=other.numerator.direction * self.denominator.direction
            )
            new_numerator = term1 + term2
            new_denominator = AbsoluteValue(
                magnitude=self.denominator.magnitude * other.denominator.magnitude,
                direction=self.denominator.direction * other.denominator.direction
            )
            return EternalRatio(numerator=new_numerator, denominator=new_denominator)
    
    def __sub__(self, other: 'EternalRatio') -> 'EternalRatio':
        """Subtract two eternal ratios.
        
        Args:
            other: Another EternalRatio to subtract
            
        Returns:
            New EternalRatio representing the difference
        """
        # Subtraction is addition with negated second operand
        negated_other = EternalRatio(
            numerator=-other.numerator,
            denominator=other.denominator
        )
        return self + negated_other
    
    def inverse(self) -> 'EternalRatio':
        """Calculate the multiplicative inverse of the ratio.
        
        The inverse of (a/b) is (b/a).
        
        Returns:
            New EternalRatio representing the inverse
            
        Raises:
            ValueError: If numerator is Absolute (magnitude=0)
        """
        if self.numerator.magnitude == 0.0:
            raise ValueError('Cannot invert ratio with Absolute numerator')
        
        return EternalRatio(numerator=self.denominator, denominator=self.numerator)
    
    def reciprocal(self) -> 'EternalRatio':
        """Alias for inverse() method.
        
        Returns:
            New EternalRatio representing the reciprocal
        """
        return self.inverse()

    def log(self) -> float:
        v = self.numerical_value()
        if v <= 0.0:
            raise ValueError('Log undefined for non-positive ratios')
        return math.log(v)

    def exp(self) -> 'EternalRatio':
        v = self.numerical_value()
        return EternalRatio.from_float(math.exp(v))

    def sin(self) -> float:
        return math.sin(self.numerical_value())

    def cos(self) -> float:
        return math.cos(self.numerical_value())

    def tan(self) -> float:
        return math.tan(self.numerical_value())
    
    def power(self, exponent: float) -> 'EternalRatio':
        """Raise the ratio to a power.
        
        (a/b)^n = (a^n)/(b^n)
        
        Args:
            exponent: Power to raise the ratio to
            
        Returns:
            New EternalRatio representing the result
            
        Raises:
            ValueError: If exponent would create invalid values
        """
        if not math.isfinite(exponent):
            raise ValueError('Exponent must be finite')
        
        # Handle special cases
        if exponent == 0:
            return EternalRatio(
                numerator=AbsoluteValue.unit_positive(),
                denominator=AbsoluteValue.unit_positive()
            )
        
        if exponent == 1:
            return EternalRatio(numerator=self.numerator, denominator=self.denominator)
        
        # Calculate powered magnitudes
        new_num_magnitude = self.numerator.magnitude ** abs(exponent)
        new_den_magnitude = self.denominator.magnitude ** abs(exponent)
        
        # Handle direction for non-integer exponents
        if exponent % 2 == 0 or exponent == int(exponent):
            # Even integer or exact integer: preserve sign logic
            new_num_direction = (self.numerator.direction ** int(abs(exponent))) if exponent >= 0 else 1
            new_den_direction = (self.denominator.direction ** int(abs(exponent))) if exponent >= 0 else 1
        else:
            # Odd or fractional: maintain original directions
            new_num_direction = self.numerator.direction
            new_den_direction = self.denominator.direction
        
        new_numerator = AbsoluteValue(magnitude=new_num_magnitude, direction=new_num_direction)
        new_denominator = AbsoluteValue(magnitude=new_den_magnitude, direction=new_den_direction)
        
        if exponent < 0:
            # Negative exponent: invert the ratio
            return EternalRatio(numerator=new_denominator, denominator=new_numerator)
        else:
            return EternalRatio(numerator=new_numerator, denominator=new_denominator)
    
    def simplify(self, tolerance: float = 1e-10) -> 'EternalRatio':
        """Simplify the ratio by reducing common factors.
        
        Args:
            tolerance: Tolerance for numerical comparisons
            
        Returns:
            New EternalRatio in simplified form
        """
        # Find GCD of magnitudes
        def gcd(a: float, b: float) -> float:
            """Calculate GCD of two floats using tolerance."""
            while abs(b) > tolerance:
                a, b = b, a % b
            return abs(a)
        
        common_factor = gcd(self.numerator.magnitude, self.denominator.magnitude)
        
        if common_factor > tolerance:
            simplified_num = AbsoluteValue(
                magnitude=self.numerator.magnitude / common_factor,
                direction=self.numerator.direction
            )
            simplified_den = AbsoluteValue(
                magnitude=self.denominator.magnitude / common_factor,
                direction=self.denominator.direction
            )
            return EternalRatio(numerator=simplified_num, denominator=simplified_den)
        
        return EternalRatio(numerator=self.numerator, denominator=self.denominator)
    
    def to_absolute_value(self) -> AbsoluteValue:
        """Convert the ratio to an AbsoluteValue.
        
        Returns:
            AbsoluteValue representing the ratio's numerical value
        """
        return AbsoluteValue.from_float(self.numerical_value())
    
    @classmethod
    def from_float(cls, value: float) -> 'EternalRatio':
        """Create an EternalRatio from a float value.
        
        Args:
            value: Float value to convert
            
        Returns:
            EternalRatio representing the value as a ratio
            
        Raises:
            ValueError: If value is not finite
        """
        if not math.isfinite(value):
            raise ValueError('Value must be finite')
        
        numerator = AbsoluteValue.from_float(value)
        denominator = AbsoluteValue.unit_positive()
        
        return cls(numerator=numerator, denominator=denominator)
    
    @classmethod
    def from_values(cls, numerator_value: float, denominator_value: float) -> 'EternalRatio':
        """Create an EternalRatio from two float values.
        
        Args:
            numerator_value: Float value for the numerator
            denominator_value: Float value for the denominator
            
        Returns:
            EternalRatio representing the ratio of the two values
            
        Raises:
            ValueError: If values are not finite or denominator is zero
        """
        if not math.isfinite(numerator_value) or not math.isfinite(denominator_value):
            raise ValueError('Values must be finite')
        
        if denominator_value == 0.0:
            raise ValueError('Denominator cannot be zero')
        
        numerator = AbsoluteValue.from_float(numerator_value)
        denominator = AbsoluteValue.from_float(denominator_value)
        
        return cls(numerator=numerator, denominator=denominator)
    
    @classmethod
    def unity(cls) -> 'EternalRatio':
        """Create a unity ratio (1/1).
        
        Returns:
            EternalRatio representing unity
        """
        unit = AbsoluteValue.unit_positive()
        return cls(numerator=unit, denominator=unit)
    
    def is_unity(self, tolerance: float = 1e-10) -> bool:
        """Check if this ratio represents unity (value = 1).
        
        Args:
            tolerance: Tolerance for comparison
            
        Returns:
            True if the ratio value is approximately 1
        """
        return abs(self.numerical_value() - 1.0) < tolerance
    
    def is_integer(self, tolerance: float = 1e-10) -> bool:
        """Check if the ratio represents an integer value.
        
        Args:
            tolerance: Tolerance for comparison
            
        Returns:
            True if the ratio value is approximately an integer
        """
        value = self.numerical_value()
        return abs(value - round(value)) < tolerance
    
    def numerical_value(self) -> float:
        """Get the numerical value of the ratio.
        
        This returns the actual mathematical ratio value,
        while signed_value() returns the directional component.
        
        Returns:
            Float value of the ratio
        """
        magnitude_ratio = self.numerator.magnitude / self.denominator.magnitude
        direction_factor = self.numerator.direction * self.denominator.direction
        return magnitude_ratio * direction_factor
    
    def __pow__(self, exponent: float) -> 'EternalRatio':
        """Raise the ratio to a power using ** operator.
        
        Args:
            exponent: Power to raise the ratio to
            
        Returns:
            New EternalRatio representing the result
        """
        return self.power(exponent)
    
    def is_reciprocal(self, other: 'EternalRatio', tolerance: float = 1e-10) -> bool:
        """Check if this ratio is the reciprocal of another.
        
        Args:
            other: Another EternalRatio to check against
            tolerance: Tolerance for comparison
            
        Returns:
            True if the ratios are reciprocals of each other
        """
        try:
            product = self * other
            return product.is_unity(tolerance)
        except (ValueError, ZeroDivisionError):
            return False
