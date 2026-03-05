# Copyright (c) 2024-2026 Tikhonov Andrey. All rights reserved.
# SPDX-License-Identifier: MIT (non-commercial) | Commercial use: see COMMERCIAL_LICENSE.md
"""AbsoluteValue class implementation for Balansis library.

This module implements the core AbsoluteValue type that replaces traditional zero
with the concept of "Absolute" - a value with magnitude and direction that enables
stable mathematical operations without division by zero issues.
"""

import math
from typing import Literal, Union, Any
from pydantic import BaseModel, validator, Field

# Type aliases for clarity
Direction = Literal[-1, 1]
Magnitude = float


class AbsoluteValue(BaseModel):
    """Core mathematical type representing values with magnitude and direction.
    
    AbsoluteValue replaces traditional zero with a stable representation that
    maintains both magnitude (non-negative) and direction (+1 or -1). This
    enables compensated operations that avoid mathematical instabilities.
    
    Attributes:
        magnitude: Non-negative float representing the size of the value
        direction: Either +1 or -1 indicating the sign/orientation
    
    Examples:
        >>> absolute_zero = AbsoluteValue(magnitude=0.0, direction=1)
        >>> positive_five = AbsoluteValue(magnitude=5.0, direction=1)
        >>> negative_three = AbsoluteValue(magnitude=3.0, direction=-1)
        >>> result = positive_five + negative_three  # Compensated addition
    """
    
    magnitude: Magnitude = Field(
        ..., 
        ge=0.0, 
        description="Non-negative magnitude of the absolute value"
    )
    direction: Direction = Field(
        ..., 
        description="Direction indicator: +1 for positive, -1 for negative"
    )
    
    class Config:
        """Pydantic configuration for AbsoluteValue."""
        frozen = True  # Make instances immutable
        validate_assignment = True
        arbitrary_types_allowed = False
        
    @validator('magnitude')
    def magnitude_must_be_finite(cls, v: float) -> float:
        """Ensure magnitude is finite and non-negative."""
        if not math.isfinite(v):
            raise ValueError('Magnitude must be finite (not inf or nan)')
        if v < 0:
            raise ValueError('Magnitude must be non-negative')
        return v
    
    @validator('direction')
    def direction_must_be_valid(cls, v: int) -> int:
        """Ensure direction is exactly +1 or -1."""
        if v not in [-1, 1]:
            raise ValueError('Direction must be exactly +1 or -1')
        return v
    
    def __add__(self, other: Any) -> 'AbsoluteValue':
        """Compensated addition following ACT principles.
        
        Rules:
        - Same direction: magnitudes add
        - Different directions: magnitudes subtract, larger direction wins
        - Equal magnitudes, different directions: result is Absolute (magnitude=0)
        
        Args:
            other: Another AbsoluteValue to add
            
        Returns:
            New AbsoluteValue representing the compensated sum
        """
        if not isinstance(other, AbsoluteValue):
            if isinstance(other, (int, float)) and math.isfinite(other):
                other = AbsoluteValue.from_float(float(other))
            else:
                return NotImplemented
        if self.direction == other.direction:
            # Same direction: magnitudes add
            return AbsoluteValue(
                magnitude=self.magnitude + other.magnitude,
                direction=self.direction
            )
        else:
            # Different directions: compensated subtraction
            if self.magnitude > other.magnitude:
                return AbsoluteValue(
                    magnitude=self.magnitude - other.magnitude,
                    direction=self.direction
                )
            elif other.magnitude > self.magnitude:
                return AbsoluteValue(
                    magnitude=other.magnitude - self.magnitude,
                    direction=other.direction
                )
            else:
                # Perfect compensation: equal magnitudes, different directions
                return AbsoluteValue(magnitude=0.0, direction=1)
    
    def __sub__(self, other: Any) -> 'AbsoluteValue':
        """Compensated subtraction.
        
        Implemented as addition with inverted direction of the second operand.
        
        Args:
            other: AbsoluteValue to subtract
            
        Returns:
            New AbsoluteValue representing the difference
        """
        if not isinstance(other, AbsoluteValue):
            if isinstance(other, (int, float)) and math.isfinite(other):
                other = AbsoluteValue.from_float(float(other))
            else:
                return NotImplemented
        inverted_other = AbsoluteValue(magnitude=other.magnitude, direction=-other.direction)
        return self + inverted_other
    
    def __mul__(self, scalar: Union[float, int]) -> 'AbsoluteValue':
        """Scalar multiplication.
        
        Args:
            scalar: Numeric value to multiply by
            
        Returns:
            New AbsoluteValue with scaled magnitude
            
        Raises:
            ValueError: If scalar is not finite
        """
        if not math.isfinite(scalar):
            raise ValueError('Scalar must be finite')
        
        new_magnitude = abs(scalar) * self.magnitude
        new_direction = self.direction if scalar >= 0 else -self.direction
        
        return AbsoluteValue(magnitude=new_magnitude, direction=new_direction)
    
    def __rmul__(self, scalar: Union[float, int]) -> 'AbsoluteValue':
        """Right scalar multiplication (commutative)."""
        return self.__mul__(scalar)

    def __radd__(self, left: Any) -> 'AbsoluteValue':
        if isinstance(left, (int, float)) and math.isfinite(left):
            return AbsoluteValue.from_float(float(left)) + self
        return NotImplemented

    def __rsub__(self, left: Any) -> 'AbsoluteValue':
        if isinstance(left, (int, float)) and math.isfinite(left):
            return AbsoluteValue.from_float(float(left)) - self
        return NotImplemented

    def __rtruediv__(self, left: Any) -> 'AbsoluteValue':
        if isinstance(left, (int, float)) and math.isfinite(left):
            return AbsoluteValue.from_float(float(left)) / self.to_float()
        return NotImplemented
    
    def __truediv__(self, scalar: Union[float, int]) -> 'AbsoluteValue':
        """Scalar division.
        
        Args:
            scalar: Non-zero numeric value to divide by
            
        Returns:
            New AbsoluteValue with scaled magnitude
            
        Raises:
            ValueError: If scalar is zero or not finite
        """
        if scalar == 0:
            raise ValueError('Cannot divide by zero')
        if not math.isfinite(scalar):
            raise ValueError('Scalar must be finite')
        
        return self.__mul__(1.0 / scalar)
    
    def __neg__(self) -> 'AbsoluteValue':
        """Unary negation (direction inversion).
        
        Returns:
            New AbsoluteValue with inverted direction
        """
        return AbsoluteValue(magnitude=self.magnitude, direction=-self.direction)
    
    def __abs__(self) -> 'AbsoluteValue':
        """Return the absolute value (positive direction).
        
        Returns:
            New AbsoluteValue with positive direction
        """
        return AbsoluteValue(magnitude=self.magnitude, direction=1)
    
    def __eq__(self, other: Any) -> bool:
        """Equality comparison.
        
        Args:
            other: Object to compare with
            
        Returns:
            True if both magnitude and direction are equal
        """
        if not isinstance(other, AbsoluteValue):
            return False
        return (self.magnitude == other.magnitude and 
                self.direction == other.direction)
    
    def __lt__(self, other: 'AbsoluteValue') -> bool:
        """Less than comparison based on signed value.
        
        Returns:
            True if this value is less than the other
        """
        return self.to_float() < other.to_float()
    
    def __le__(self, other: 'AbsoluteValue') -> bool:
        """Less than or equal comparison."""
        return self.to_float() <= other.to_float()
    
    def __gt__(self, other: 'AbsoluteValue') -> bool:
        """Greater than comparison based on signed value.
        
        Returns:
            True if this value is greater than the other
        """
        return self.to_float() > other.to_float()
    
    def __ge__(self, other: 'AbsoluteValue') -> bool:
        """Greater than or equal comparison."""
        return self.to_float() >= other.to_float()
    
    def __hash__(self) -> int:
        """Hash function for use in sets and dictionaries."""
        return hash((self.magnitude, self.direction))
    
    def __repr__(self) -> str:
        """Return detailed string representation."""
        if self.is_absolute():
            return "AbsoluteValue(magnitude=0.0, direction=1)"
        return f"AbsoluteValue(magnitude={self.magnitude}, direction={self.direction:.1f})"
    
    def __str__(self) -> str:
        """Return string representation."""
        if self.is_absolute():
            return "AbsoluteValue(Absolute)"
        direction_str = "+" if self.direction > 0 else "-"
        return f"AbsoluteValue({self.magnitude}, {direction_str})"

    def __getstate__(self):
        return {"magnitude": self.magnitude, "direction": self.direction}

    def __setstate__(self, state):
        object.__setattr__(self, "magnitude", state["magnitude"])
        object.__setattr__(self, "direction", state["direction"])

    def to_json(self):
        return {"type": "AbsoluteValue", "magnitude": self.magnitude, "direction": self.direction}

    @classmethod
    def __get_validators__(cls):
        def validate(v):
            if isinstance(v, AbsoluteValue):
                return v
            if isinstance(v, (int, float)):
                return AbsoluteValue.from_float(float(v))
            if isinstance(v, dict) and "magnitude" in v and "direction" in v:
                return AbsoluteValue(magnitude=float(v["magnitude"]), direction=int(v["direction"]))
            raise TypeError("invalid AbsoluteValue input")
        yield validate
    
    def inverse(self) -> 'AbsoluteValue':
        """Calculate the multiplicative inverse.
        
        For AbsoluteValue, the inverse maintains the same direction
        but inverts the magnitude (1/magnitude).
        
        Returns:
            New AbsoluteValue representing the inverse
            
        Raises:
            ValueError: If magnitude is zero (Absolute has no inverse)
        """
        if self.magnitude == 0.0:
            raise ValueError('Absolute (magnitude=0) has no multiplicative inverse')
        
        return AbsoluteValue(
            magnitude=1.0 / self.magnitude,
            direction=self.direction
        )

    def log(self) -> float:
        v = self.to_float()
        if v <= 0.0:
            raise ValueError('Log undefined for non-positive values')
        return math.log(v)

    def exp(self) -> 'AbsoluteValue':
        v = self.to_float()
        return AbsoluteValue(magnitude=math.exp(abs(v)), direction=1)

    def sin(self) -> float:
        return math.sin(self.to_float())

    def cos(self) -> float:
        return math.cos(self.to_float())

    def tan(self) -> float:
        return math.tan(self.to_float())
    
    def to_float(self) -> float:
        """Convert to standard Python float.
        
        Returns:
            Float representation: magnitude * direction
        """
        return self.magnitude * self.direction
    
    def is_absolute(self) -> bool:
        """Check if this value represents Absolute (magnitude=0).
        
        Returns:
            True if magnitude is zero
        """
        return self.magnitude == 0.0
    
    def is_positive(self) -> bool:
        """Check if this value is positive.
        
        Returns:
            True if direction is +1 and magnitude > 0
        """
        return self.direction == 1 and self.magnitude > 0
    
    def is_negative(self) -> bool:
        """Check if this value is negative.
        
        Returns:
            True if direction is -1 and magnitude > 0
        """
        return self.direction == -1 and self.magnitude > 0
    
    def is_unit(self) -> bool:
        """Check if this value is a unit (magnitude=1.0).
        
        Returns:
            True if magnitude is 1.0
        """
        return self.magnitude == 1.0
    
    def is_compensating(self) -> bool:
        """
        Check if this value is compensating according to ACT theory.
        
        A value is compensating if it's either absolute or has a very large magnitude
        that triggers compensation behavior.
        
        Returns:
            bool: True if the value is compensating, False otherwise.
        """
        if self.is_absolute():
            return True
        # Large magnitude values are considered compensating
        return self.magnitude >= 1e10
    
    def compensates_with(self, other: 'AbsoluteValue') -> bool:
        """Check if this value perfectly compensates with another.
        
        Perfect compensation occurs when two values have equal magnitudes
        but opposite directions, resulting in Absolute when added.
        
        Args:
            other: Another AbsoluteValue to check compensation with
            
        Returns:
            True if the values perfectly compensate
        """
        return (self.magnitude == other.magnitude and 
                self.direction != other.direction)
    
    @classmethod
    def from_float(cls, value: float) -> 'AbsoluteValue':
        """Create AbsoluteValue from a standard float.
        
        Args:
            value: Float value to convert
            
        Returns:
            New AbsoluteValue with appropriate magnitude and direction
            
        Raises:
            ValueError: If value is not finite
        """
        if not math.isfinite(value):
            raise ValueError('Value must be finite')
        
        magnitude = abs(value)
        direction = 1 if value >= 0 else -1
        
        return cls(magnitude=magnitude, direction=direction)
    
    @classmethod
    def absolute(cls) -> 'AbsoluteValue':
        """Create the Absolute value (magnitude=0, direction=+1).
        
        Returns:
            AbsoluteValue representing Absolute
        """
        return cls(magnitude=0.0, direction=1)
    
    @classmethod
    def unit_positive(cls) -> 'AbsoluteValue':
        """Create a positive unit value.
        
        Returns:
            AbsoluteValue with magnitude=1.0, direction=+1
        """
        return cls(magnitude=1.0, direction=1)
    
    @classmethod
    def unit_negative(cls) -> 'AbsoluteValue':
        """Create a negative unit value.
        
        Returns:
            AbsoluteValue with magnitude=1.0, direction=-1
        """
        return cls(magnitude=1.0, direction=-1)
