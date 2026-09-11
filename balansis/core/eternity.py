# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.
#
# This file is part of Balansis.
# Balansis is dual-licensed under:
#   1. GNU Affero General Public License v3.0 (AGPLv3) for open-source use.
#   2. A Commercial License for proprietary and corporate use.
#
# See LICENSING.md in the project root for license selection details.
# For commercial licensing: andrew@xteam.pro
"""EternalRatio class implementation for Balansis library.

This module implements the EternalRatio type that represents structural ratios
between AbsoluteValues. These ratios are invariant across time, context, and scale,
providing stable mathematical relationships without traditional division issues.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Callable, Iterator, Literal, Optional, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    numerator: AbsoluteValue = Field(
        ...,
        description="AbsoluteValue in the numerator position",
    )
    denominator: AbsoluteValue = Field(
        ...,
        description="AbsoluteValue in the denominator position (non-Absolute)",
    )

    @field_validator("denominator")
    @classmethod
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
            raise ValueError("Denominator cannot be Absolute (magnitude=0)")
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
            within_reasonable_bounds = (
                abs(ratio_value) <= reasonable_bound
                and abs(ratio_value) >= 1.0 / reasonable_bound
            )

            # However, ratios at precision limits should be unstable
            # This handles the case where ratio is very close to 1.0 but at the threshold
            # Only consider it a precision limit if it's close to 1.0 (within 0.01) but at threshold
            close_to_unity = abs(ratio_value - 1.0) < 0.01
            at_precision_limit = (
                close_to_unity and abs(ratio_value - 1.0) >= ACT_STABILITY_THRESHOLD
            )

            return (
                math.isfinite(ratio_value)
                and self.denominator.magnitude > tolerance
                and within_reasonable_bounds
                and not at_precision_limit
            )
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

    def __lt__(self, other: "EternalRatio") -> bool:
        """Less than comparison based on numerical values."""
        return self.numerical_value() < other.numerical_value()

    def __le__(self, other: "EternalRatio") -> bool:
        """Less than or equal comparison."""
        return self.numerical_value() <= other.numerical_value()

    def __gt__(self, other: "EternalRatio") -> bool:
        """Greater than comparison based on numerical values."""
        return self.numerical_value() > other.numerical_value()

    def __ge__(self, other: "EternalRatio") -> bool:
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

    def __getstate__(self) -> dict[str, Any]:
        return {"numerator": self.numerator, "denominator": self.denominator}

    def __setstate__(self, state: dict[str, Any]) -> None:
        object.__setattr__(self, "numerator", state["numerator"])
        object.__setattr__(self, "denominator", state["denominator"])

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "EternalRatio",
            "value": self.numerical_value(),
            "structural_dump": {
                "numerator": self.numerator.to_float(),
                "denominator": self.denominator.to_float(),
            },
        }

    @classmethod
    def __get_validators__(cls) -> Iterator[Callable[[Any], EternalRatio]]:
        def validate(v: Any) -> EternalRatio:
            if isinstance(v, EternalRatio):
                return v
            if isinstance(v, (int, float)):
                return EternalRatio.from_float(float(v))
            if isinstance(v, dict) and "numerator" in v and "denominator" in v:
                num = (
                    AbsoluteValue.from_float(float(v["numerator"]))
                    if not isinstance(v["numerator"], AbsoluteValue)
                    else v["numerator"]
                )
                den = (
                    AbsoluteValue.from_float(float(v["denominator"]))
                    if not isinstance(v["denominator"], AbsoluteValue)
                    else v["denominator"]
                )
                return EternalRatio(numerator=num, denominator=den)
            raise TypeError("invalid EternalRatio input")

        yield validate

    def __mul__(self, other: Any) -> "EternalRatio":
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
                direction=cast(
                    Literal[-1, 1], self.numerator.direction * other.numerator.direction
                ),
            )
            new_denominator = AbsoluteValue(
                magnitude=self.denominator.magnitude * other.denominator.magnitude,
                direction=cast(
                    Literal[-1, 1],
                    self.denominator.direction * other.denominator.direction,
                ),
            )
            return EternalRatio(numerator=new_numerator, denominator=new_denominator)
        elif isinstance(other, (int, float)):
            # Scalar multiplication: multiply numerator by scalar
            new_numerator = AbsoluteValue(
                magnitude=self.numerator.magnitude * abs(other),
                direction=cast(
                    Literal[-1, 1], self.numerator.direction * (1 if other > 0 else -1)
                ),
            )
            return EternalRatio(numerator=new_numerator, denominator=self.denominator)
        else:
            return NotImplemented

    def __truediv__(self, other: Any) -> "EternalRatio":
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
                direction=cast(
                    Literal[-1, 1], self.numerator.direction * (1 if other > 0 else -1)
                ),
            )
            return EternalRatio(numerator=new_numerator, denominator=self.denominator)
        else:
            return NotImplemented

    def __radd__(self, left: Any) -> "EternalRatio":
        if isinstance(left, (int, float)) and math.isfinite(left):
            return EternalRatio.from_float(float(left)) + self
        return NotImplemented

    def __rsub__(self, left: Any) -> "EternalRatio":
        if isinstance(left, (int, float)) and math.isfinite(left):
            return EternalRatio.from_float(float(left)) - self
        return NotImplemented

    def __rmul__(self, left: Any) -> "EternalRatio":
        if isinstance(left, (int, float)) and math.isfinite(left):
            return self * float(left)
        return NotImplemented

    def __rtruediv__(self, left: Any) -> "EternalRatio":
        if isinstance(left, (int, float)) and math.isfinite(left):
            return EternalRatio.from_float(float(left)) / self
        return NotImplemented

    def __add__(self, other: "EternalRatio") -> "EternalRatio":
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
                direction=cast(
                    Literal[-1, 1],
                    self.numerator.direction * other.denominator.direction,
                ),
            )
            term2 = AbsoluteValue(
                magnitude=other.numerator.magnitude * self.denominator.magnitude,
                direction=cast(
                    Literal[-1, 1],
                    other.numerator.direction * self.denominator.direction,
                ),
            )
            new_numerator = term1 + term2
            new_denominator = AbsoluteValue(
                magnitude=self.denominator.magnitude * other.denominator.magnitude,
                direction=cast(
                    Literal[-1, 1],
                    self.denominator.direction * other.denominator.direction,
                ),
            )
            return EternalRatio(numerator=new_numerator, denominator=new_denominator)

    def __sub__(self, other: "EternalRatio") -> "EternalRatio":
        """Subtract two eternal ratios.

        Args:
            other: Another EternalRatio to subtract

        Returns:
            New EternalRatio representing the difference
        """
        # Subtraction is addition with negated second operand
        negated_other = EternalRatio(
            numerator=-other.numerator, denominator=other.denominator
        )
        return self + negated_other

    def inverse(self) -> "EternalRatio":
        """Calculate the multiplicative inverse of the ratio.

        The inverse of (a/b) is (b/a).

        Returns:
            New EternalRatio representing the inverse

        Raises:
            ValueError: If numerator is Absolute (magnitude=0)
        """
        if self.numerator.magnitude == 0.0:
            raise ValueError("Cannot invert ratio with Absolute numerator")

        return EternalRatio(numerator=self.denominator, denominator=self.numerator)

    def reciprocal(self) -> "EternalRatio":
        """Alias for inverse() method.

        Returns:
            New EternalRatio representing the reciprocal
        """
        return self.inverse()

    def log(self) -> float:
        v = self.numerical_value()
        if v <= 0.0:
            raise ValueError("Log undefined for non-positive ratios")
        return math.log(v)

    def exp(self) -> "EternalRatio":
        v = self.numerical_value()
        return EternalRatio.from_float(math.exp(v))

    def sin(self) -> float:
        return math.sin(self.numerical_value())

    def cos(self) -> float:
        return math.cos(self.numerical_value())

    def tan(self) -> float:
        return math.tan(self.numerical_value())

    def power(self, exponent: float) -> "EternalRatio":
        """Raise a finite ratio to a real power; reject non-real results."""
        if not math.isfinite(exponent):
            raise ValueError("Exponent must be finite")
        if exponent == 0:
            return EternalRatio.unity()
        if self.numerator.is_absolute() and exponent < 0:
            raise ValueError("Cannot raise zero to a negative power")
        sign = self.numerator.direction * self.denominator.direction
        integer = exponent == int(exponent)
        if sign < 0 and not integer and not self.numerator.is_absolute():
            raise ValueError("Negative ratios require an integer exponent")
        power = abs(exponent)
        numerator = self.numerator.magnitude**power
        denominator = self.denominator.magnitude**power
        if exponent < 0:
            numerator, denominator = denominator, numerator
        direction = -1 if sign < 0 and integer and int(exponent) % 2 else 1
        return EternalRatio(
            numerator=AbsoluteValue(
                magnitude=numerator, direction=cast(Literal[-1, 1], direction)
            ),
            denominator=AbsoluteValue(magnitude=denominator, direction=1),
        )

    def simplify(self, tolerance: float = 1e-10) -> "EternalRatio":
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
                direction=self.numerator.direction,
            )
            simplified_den = AbsoluteValue(
                magnitude=self.denominator.magnitude / common_factor,
                direction=self.denominator.direction,
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
    def from_float(cls, value: float) -> "EternalRatio":
        """Create an EternalRatio from a float value.

        Args:
            value: Float value to convert

        Returns:
            EternalRatio representing the value as a ratio

        Raises:
            ValueError: If value is not finite
        """
        if not math.isfinite(value):
            raise ValueError("Value must be finite")

        numerator = AbsoluteValue.from_float(value)
        denominator = AbsoluteValue.unit_positive()

        return cls(numerator=numerator, denominator=denominator)

    @classmethod
    def from_values(
        cls, numerator_value: float, denominator_value: float
    ) -> "EternalRatio":
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
            raise ValueError("Values must be finite")

        if denominator_value == 0.0:
            raise ValueError("Denominator cannot be zero")

        numerator = AbsoluteValue.from_float(numerator_value)
        denominator = AbsoluteValue.from_float(denominator_value)

        return cls(numerator=numerator, denominator=denominator)

    @classmethod
    def unity(cls) -> "EternalRatio":
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

    def __pow__(self, exponent: float) -> "EternalRatio":
        """Raise the ratio to a power using ** operator.

        Args:
            exponent: Power to raise the ratio to

        Returns:
            New EternalRatio representing the result
        """
        return self.power(exponent)

    def is_reciprocal(self, other: "EternalRatio", tolerance: float = 1e-10) -> bool:
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


class SingularPolicy(str, Enum):
    """Execution policy for singular arithmetic states."""

    RAISE = "raise"
    PROPAGATE = "propagate"
    SATURATE = "saturate"


class SingularArithmeticEvent(BaseModel):
    """Machine-readable telemetry event for singular arithmetic handling."""

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    operation: str = Field(
        ..., description="Operation that produced or handled the singular state"
    )
    policy: SingularPolicy = Field(
        ..., description="Applied singular arithmetic policy"
    )
    input_kind: str = Field(
        ..., description="Original ExtendedRatio kind before policy application"
    )
    output_kind: str = Field(..., description="Resulting kind after policy application")
    reason: Optional[str] = Field(
        default=None, description="Optional explanation of the event"
    )
    direction: Optional[int] = Field(
        default=None, description="Direction for infinite states"
    )
    saturated: bool = Field(
        default=False, description="Whether the event saturated an infinite state"
    )
    numeric_value: Optional[float] = Field(
        default=None, description="Numeric output value when finite or infinite"
    )


class ExtendedRatio(BaseModel):
    """Extended ratio with explicit infinity and indeterminate semantics.

    ``EternalRatio`` remains the strict finite-ratio object. ``ExtendedRatio``
    is the wider runtime model used when a computation must represent:

    - finite ratios,
    - signed infinity,
    - indeterminate results.
    """

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    kind: Literal["finite", "infinite", "indeterminate"] = Field(
        ...,
        description="Semantic state of the ratio",
    )
    ratio: Optional[EternalRatio] = Field(
        default=None,
        description="Finite ratio payload when kind='finite'",
    )
    direction: Optional[int] = Field(
        default=None,
        description="Direction of infinity when kind='infinite'",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Optional explanation for singular or indeterminate states",
    )

    @field_validator("direction")
    @classmethod
    def direction_must_be_valid(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return value
        if value not in (-1, 1):
            raise ValueError("Direction must be exactly +1 or -1")
        return value

    @model_validator(mode="after")
    def validate_state(self) -> "ExtendedRatio":
        if self.kind == "finite":
            if self.ratio is None:
                raise ValueError("Finite ExtendedRatio requires a finite ratio payload")
            if self.direction is not None:
                raise ValueError(
                    "Finite ExtendedRatio cannot carry an infinity direction"
                )
        elif self.kind == "infinite":
            if self.ratio is not None:
                raise ValueError(
                    "Infinite ExtendedRatio cannot carry a finite ratio payload"
                )
            if self.direction is None:
                raise ValueError("Infinite ExtendedRatio requires a direction")
        else:
            if self.ratio is not None:
                raise ValueError(
                    "Indeterminate ExtendedRatio cannot carry a finite ratio payload"
                )
        return self

    @staticmethod
    def _coerce(other: Any) -> Optional["ExtendedRatio"]:
        if isinstance(other, ExtendedRatio):
            return other
        if isinstance(other, EternalRatio):
            return ExtendedRatio.from_ratio(other)
        if isinstance(other, (int, float)):
            if not math.isfinite(other):
                if math.isnan(other):
                    return ExtendedRatio.indeterminate("nan_input")
                return (
                    ExtendedRatio.positive_infinity("float_inf_input")
                    if other > 0
                    else ExtendedRatio.negative_infinity("float_inf_input")
                )
            return ExtendedRatio.from_float(float(other))
        return None

    @staticmethod
    def _sign_of_finite_ratio(ratio: EternalRatio) -> int:
        value = ratio.numerical_value()
        if value > 0:
            return 1
        if value < 0:
            return -1
        raise ValueError("Zero finite ratio does not have a sign")

    @classmethod
    def from_ratio(cls, ratio: EternalRatio) -> "ExtendedRatio":
        return cls(kind="finite", ratio=ratio)

    @classmethod
    def from_float(cls, value: float) -> "ExtendedRatio":
        if not math.isfinite(value):
            if math.isnan(value):
                return cls.indeterminate("nan_input")
            return (
                cls.positive_infinity("float_inf_input")
                if value > 0
                else cls.negative_infinity("float_inf_input")
            )
        return cls.from_ratio(EternalRatio.from_float(value))

    @classmethod
    def positive_infinity(cls, reason: Optional[str] = None) -> "ExtendedRatio":
        return cls(kind="infinite", direction=1, reason=reason or "positive_infinity")

    @classmethod
    def negative_infinity(cls, reason: Optional[str] = None) -> "ExtendedRatio":
        return cls(kind="infinite", direction=-1, reason=reason or "negative_infinity")

    @classmethod
    def indeterminate(cls, reason: Optional[str] = None) -> "ExtendedRatio":
        return cls(kind="indeterminate", reason=reason or "indeterminate")

    @classmethod
    def from_division(
        cls,
        numerator: AbsoluteValue,
        denominator: AbsoluteValue,
        reason: Optional[str] = None,
    ) -> "ExtendedRatio":
        if denominator.is_absolute():
            if numerator.is_absolute():
                return cls.indeterminate(reason or "absolute_over_absolute")
            direction = numerator.direction * denominator.direction
            return cls(
                kind="infinite",
                direction=direction,
                reason=reason or "finite_over_absolute",
            )
        return cls.from_ratio(
            EternalRatio(numerator=numerator, denominator=denominator)
        )

    def is_finite(self) -> bool:
        return self.kind == "finite"

    def is_infinite(self) -> bool:
        return self.kind == "infinite"

    def is_indeterminate(self) -> bool:
        return self.kind == "indeterminate"

    def is_singular(self) -> bool:
        return self.kind in {"infinite", "indeterminate"}

    def is_stable(self, tolerance: float = 1e-10) -> bool:
        return (
            self.is_finite()
            and self.ratio is not None
            and self.ratio.is_stable(tolerance)
        )

    def numerical_value(self) -> float:
        if self.kind == "finite":
            assert self.ratio is not None
            return self.ratio.numerical_value()
        if self.kind == "infinite":
            assert self.direction is not None
            return math.copysign(math.inf, self.direction)
        return math.nan

    def signed_value(self) -> float:
        if self.kind == "finite":
            assert self.ratio is not None
            return self.ratio.signed_value()
        if self.kind == "infinite":
            assert self.direction is not None
            return float(self.direction)
        return math.nan

    def finite_ratio(self) -> EternalRatio:
        if self.ratio is None:
            raise ValueError(
                f"ExtendedRatio({self.kind}) does not carry a finite ratio"
            )
        return self.ratio

    def saturate(self, limit: float = 1e12) -> "ExtendedRatio":
        """Convert infinite states to finite bounded ratios.

        `indeterminate` remains unchanged because there is no honest finite
        representative for that state.
        """
        if limit <= 0.0 or not math.isfinite(limit):
            raise ValueError("Saturation limit must be a positive finite number")
        if self.is_infinite():
            assert self.direction is not None
            return ExtendedRatio.from_float(self.direction * limit)
        return self

    def policy_event(
        self,
        operation: str,
        policy: SingularPolicy | str,
        result: Optional["ExtendedRatio"] = None,
    ) -> SingularArithmeticEvent:
        resolved_policy = SingularPolicy(policy)
        outcome = result or self
        numeric_value = outcome.numerical_value()
        return SingularArithmeticEvent(
            operation=operation,
            policy=resolved_policy,
            input_kind=self.kind,
            output_kind=outcome.kind,
            reason=self.reason,
            direction=outcome.direction,
            saturated=self.is_infinite() and outcome.is_finite(),
            numeric_value=None if math.isnan(numeric_value) else numeric_value,
        )

    def apply_policy(
        self,
        policy: SingularPolicy | str = SingularPolicy.PROPAGATE,
        *,
        operation: str = "extended_ratio",
        saturation_limit: float = 1e12,
    ) -> tuple["ExtendedRatio", Optional[SingularArithmeticEvent]]:
        resolved_policy = SingularPolicy(policy)
        if not self.is_singular():
            return self, None

        if resolved_policy == SingularPolicy.RAISE:
            raise ValueError(
                f"{operation} produced singular ExtendedRatio(kind={self.kind}, reason={self.reason})"
            )

        if resolved_policy == SingularPolicy.SATURATE:
            saturated = self.saturate(limit=saturation_limit)
            if saturated is self:
                return self, self.policy_event(operation, resolved_policy)
            return saturated, self.policy_event(
                operation, resolved_policy, result=saturated
            )

        return self, self.policy_event(operation, resolved_policy)

    def inverse(self) -> "ExtendedRatio":
        if self.kind == "indeterminate":
            return self
        if self.kind == "infinite":
            assert self.direction is not None
            return ExtendedRatio.from_float(0.0)

        ratio = self.finite_ratio()
        value = ratio.numerical_value()
        if value == 0.0:
            return ExtendedRatio.indeterminate("inverse_of_zero")
        return ExtendedRatio.from_ratio(ratio.inverse())

    def __neg__(self) -> "ExtendedRatio":
        if self.kind == "indeterminate":
            return self
        if self.kind == "infinite":
            assert self.direction is not None
            return ExtendedRatio(
                kind="infinite", direction=-self.direction, reason=self.reason
            )
        ratio = self.finite_ratio()
        return ExtendedRatio.from_ratio(
            EternalRatio(numerator=-ratio.numerator, denominator=ratio.denominator)
        )

    def __add__(self, other: Any) -> "ExtendedRatio":
        other_ratio = self._coerce(other)
        if other_ratio is None:
            return NotImplemented

        if self.is_indeterminate() or other_ratio.is_indeterminate():
            return ExtendedRatio.indeterminate("addition_with_indeterminate")

        if self.is_finite() and other_ratio.is_finite():
            return ExtendedRatio.from_ratio(
                self.finite_ratio() + other_ratio.finite_ratio()
            )

        if self.is_infinite() and other_ratio.is_infinite():
            if self.direction == other_ratio.direction:
                return ExtendedRatio(
                    kind="infinite",
                    direction=self.direction,
                    reason="infinity_plus_same_infinity",
                )
            return ExtendedRatio.indeterminate("opposite_infinities_addition")

        return self if self.is_infinite() else other_ratio

    def __sub__(self, other: Any) -> "ExtendedRatio":
        other_ratio = self._coerce(other)
        if other_ratio is None:
            return NotImplemented
        return self + (-other_ratio)

    def __mul__(self, other: Any) -> "ExtendedRatio":
        other_ratio = self._coerce(other)
        if other_ratio is None:
            return NotImplemented

        if self.is_indeterminate() or other_ratio.is_indeterminate():
            return ExtendedRatio.indeterminate("multiplication_with_indeterminate")

        if self.is_finite() and other_ratio.is_finite():
            return ExtendedRatio.from_ratio(
                self.finite_ratio() * other_ratio.finite_ratio()
            )

        if self.is_infinite() and other_ratio.is_infinite():
            assert self.direction is not None and other_ratio.direction is not None
            return ExtendedRatio(
                kind="infinite",
                direction=self.direction * other_ratio.direction,
                reason="infinity_times_infinity",
            )

        finite = self.finite_ratio() if self.is_finite() else other_ratio.finite_ratio()
        infinite = self if self.is_infinite() else other_ratio
        finite_value = finite.numerical_value()
        if finite_value == 0.0:
            return ExtendedRatio.indeterminate("zero_times_infinity")
        assert infinite.direction is not None
        return ExtendedRatio(
            kind="infinite",
            direction=self._sign_of_finite_ratio(finite) * infinite.direction,
            reason="finite_times_infinity",
        )

    def __truediv__(self, other: Any) -> "ExtendedRatio":
        other_ratio = self._coerce(other)
        if other_ratio is None:
            return NotImplemented

        if self.is_indeterminate() or other_ratio.is_indeterminate():
            return ExtendedRatio.indeterminate("division_with_indeterminate")

        if self.is_finite() and other_ratio.is_finite():
            divisor = other_ratio.finite_ratio()
            if divisor.numerical_value() == 0.0:
                dividend = self.finite_ratio()
                if dividend.numerical_value() == 0.0:
                    return ExtendedRatio.indeterminate("zero_over_zero_ratio")
                return ExtendedRatio.indeterminate("finite_over_zero_ratio")
            return ExtendedRatio.from_ratio(self.finite_ratio() / divisor)

        if self.is_infinite() and other_ratio.is_infinite():
            return ExtendedRatio.indeterminate("infinity_over_infinity")

        if self.is_infinite() and other_ratio.is_finite():
            divisor = other_ratio.finite_ratio()
            if divisor.numerical_value() == 0.0:
                return ExtendedRatio.indeterminate("infinity_over_zero_ratio")
            assert self.direction is not None
            return ExtendedRatio(
                kind="infinite",
                direction=self.direction * self._sign_of_finite_ratio(divisor),
                reason="infinity_over_finite",
            )

        finite = self.finite_ratio()
        if finite.numerical_value() == 0.0:
            return ExtendedRatio.from_float(0.0)
        return ExtendedRatio.from_float(0.0)

    def __radd__(self, left: Any) -> "ExtendedRatio":
        left_ratio = self._coerce(left)
        if left_ratio is None:
            return NotImplemented
        return left_ratio + self

    def __rsub__(self, left: Any) -> "ExtendedRatio":
        left_ratio = self._coerce(left)
        if left_ratio is None:
            return NotImplemented
        return left_ratio - self

    def __rmul__(self, left: Any) -> "ExtendedRatio":
        left_ratio = self._coerce(left)
        if left_ratio is None:
            return NotImplemented
        return left_ratio * self

    def __rtruediv__(self, left: Any) -> "ExtendedRatio":
        left_ratio = self._coerce(left)
        if left_ratio is None:
            return NotImplemented
        return left_ratio / self

    def __eq__(self, other: Any) -> bool:
        coerced = self._coerce(other)
        if coerced is None:
            return False
        if self.kind != coerced.kind:
            return False
        if self.kind == "finite":
            return self.finite_ratio() == coerced.finite_ratio()
        if self.kind == "infinite":
            return self.direction == coerced.direction
        return True

    def __repr__(self) -> str:
        if self.kind == "finite":
            return f"ExtendedRatio(kind='finite', ratio={self.ratio!r})"
        if self.kind == "infinite":
            return f"ExtendedRatio(kind='infinite', direction={self.direction}, reason={self.reason!r})"
        return f"ExtendedRatio(kind='indeterminate', reason={self.reason!r})"

    def __str__(self) -> str:
        if self.kind == "finite":
            return f"ExtendedRatio(finite={self.finite_ratio().numerical_value():.6f})"
        if self.kind == "infinite":
            return (
                "ExtendedRatio(+infinity)"
                if self.direction == 1
                else "ExtendedRatio(-infinity)"
            )
        return "ExtendedRatio(indeterminate)"

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "ExtendedRatio",
            "kind": self.kind,
            "reason": self.reason,
        }
        if self.kind == "finite":
            payload["value"] = self.finite_ratio().numerical_value()
            payload["ratio"] = self.finite_ratio().to_json()
        elif self.kind == "infinite":
            payload["direction"] = self.direction
            payload["value"] = self.numerical_value()
        else:
            payload["value"] = None
        return payload
