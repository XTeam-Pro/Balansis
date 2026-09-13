# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.
#
# This file is part of Balansis.
# Balansis is dual-licensed under:
#   1. GNU Affero General Public License v3.0 (AGPLv3) for open-source use.
#   2. A Commercial License for proprietary and corporate use.
#
# See LICENSING.md in the project root for license selection details.
# For commercial licensing: andrew@xteam.pro
"""Core operations module for Balansis library.

This module implements compensated arithmetic operations that maintain mathematical
stability and avoid traditional issues with zero division and infinity. All operations
are designed around the Absolute Compensation Theory (ACT) principles.
"""

import math
from typing import List, Literal, Optional, Tuple, Union, cast

from .absolute import AbsoluteValue
from .eternity import (
    EternalRatio,
    ExtendedRatio,
    SingularArithmeticEvent,
    SingularPolicy,
)

# Set high precision for decimal operations

# Type aliases for clarity
NumericType = Union[float, int, AbsoluteValue, EternalRatio, ExtendedRatio]
CompensatedResult = Tuple[AbsoluteValue, float]  # (result, compensation_factor)
CompensatedDivideResult = Tuple[EternalRatio, float]  # (ratio, compensation_factor)
ExtendedDivideResult = Tuple[
    ExtendedRatio, float
]  # (extended ratio, compensation_factor)
PolicyDivideResult = Tuple[ExtendedRatio, float, Optional[SingularArithmeticEvent]]


class Operations:
    """Core operations implementing Absolute Compensation Theory.

    Static methods for performing compensated arithmetic operations that
    maintain stability and avoid mathematical singularities. All operations
    follow ACT principles of compensation, stability, and eternity.

    Returned compensation factors:
    - 1.0 means the operation completed without applied compensation (identity).
    - >1.0 indicates accumulated compensation (overflow, Kahan correction, etc.).
    - <1.0 indicates near-Absolute / underflow detection.
    - 0.0 indicates the result was forced to Absolute (multiplicative identity broken).
    """

    # Mathematical constants for ACT
    COMPENSATION_THRESHOLD = 1e-15
    STABILITY_FACTOR = 1e-12
    OVERFLOW_THRESHOLD = 1e100
    MAX_COMPENSATION_ITERATIONS = 100

    @staticmethod
    def compensated_add(
        a: AbsoluteValue, b: AbsoluteValue, compensation_factor: float = 1.0
    ) -> CompensatedResult:
        """Perform compensated addition of two AbsoluteValues.

        Opposite operands with equal represented magnitudes cancel to Absolute.
        Distinct, nearly equal magnitudes retain their represented difference.
        Information rounded away before construction cannot be recovered here.

        The compensation factor is a diagnostic, not an additive correction or
        a rigorous error bound. Exact cancellation retains the
        legacy ``compensation_factor * STABILITY_FACTOR`` diagnostic; nonzero
        near-cancellation scales it by the ratio of magnitude to difference.
        """
        # Detect catastrophic cancellation: opposite directions and close magnitudes.
        if a.direction != b.direction and a.magnitude > 0 and b.magnitude > 0:
            larger = max(a.magnitude, b.magnitude)
            diff = abs(a.magnitude - b.magnitude)
            if a.magnitude == b.magnitude:
                return (
                    AbsoluteValue.absolute(),
                    compensation_factor * Operations.STABILITY_FACTOR,
                )
            # Near-cancellation is only meaningful when both operands are far above
            # the compensation threshold. Otherwise we treat the result as Absolute.
            if larger >= 1.0 and diff <= Operations.COMPENSATION_THRESHOLD * larger:
                residual = diff
                winner = a.direction if a.magnitude >= b.magnitude else b.direction
                compensation = compensation_factor * (
                    larger / max(residual, Operations.COMPENSATION_THRESHOLD)
                )
                return AbsoluteValue(magnitude=residual, direction=winner), compensation

        # Standard addition
        result = a + b

        # Result genuinely underflowed
        if (
            result.magnitude < Operations.COMPENSATION_THRESHOLD
            and result.magnitude > 0.0
        ):
            compensated_result = AbsoluteValue.absolute()
            applied_compensation = result.magnitude / Operations.COMPENSATION_THRESHOLD
            return compensated_result, applied_compensation

        return result, 1.0

    @staticmethod
    def compensated_multiply(
        a: AbsoluteValue, b: AbsoluteValue, compensation_factor: float = 1.0
    ) -> CompensatedResult:
        """Perform compensated multiplication of two AbsoluteValues."""
        if a.is_absolute() or b.is_absolute():
            return AbsoluteValue.absolute(), 0.0

        try:
            result_magnitude = a.magnitude * b.magnitude
            if math.isinf(result_magnitude):
                # Both factors finite but product overflowed float64 range
                log_compensation = (
                    math.log10(a.magnitude) + math.log10(b.magnitude) - 100.0
                )
                return (
                    AbsoluteValue(
                        magnitude=Operations.OVERFLOW_THRESHOLD,
                        direction=cast(Literal[-1, 1], a.direction * b.direction),
                    ),
                    log_compensation,
                )
        except OverflowError:
            log_compensation = math.log10(a.magnitude) + math.log10(b.magnitude) - 100.0
            return (
                AbsoluteValue(
                    magnitude=Operations.OVERFLOW_THRESHOLD,
                    direction=cast(Literal[-1, 1], a.direction * b.direction),
                ),
                log_compensation,
            )

        result_direction = a.direction * b.direction

        if result_magnitude > Operations.OVERFLOW_THRESHOLD:
            log_compensation = math.log10(result_magnitude) - 100.0
            return (
                AbsoluteValue(
                    magnitude=Operations.OVERFLOW_THRESHOLD,
                    direction=cast(Literal[-1, 1], result_direction),
                ),
                log_compensation,
            )

        if result_magnitude < Operations.COMPENSATION_THRESHOLD:
            # Underflow: preserve the lost magnitude in the compensation factor
            return AbsoluteValue.absolute(), result_magnitude

        return (
            AbsoluteValue(
                magnitude=result_magnitude,
                direction=cast(Literal[-1, 1], result_direction),
            ),
            1.0,
        )

    @staticmethod
    def compensated_divide(
        numerator: AbsoluteValue,
        denominator: AbsoluteValue,
        compensation_factor: float = 1.0,
    ) -> CompensatedDivideResult:
        """Perform compensated division using EternalRatio."""
        if denominator.is_absolute():
            raise ValueError("Cannot divide by Absolute (denominator magnitude=0)")

        applied_compensation = 1.0
        if denominator.magnitude < Operations.COMPENSATION_THRESHOLD:
            applied_compensation = (
                denominator.magnitude / Operations.COMPENSATION_THRESHOLD
            )

        return (
            EternalRatio(numerator=numerator, denominator=denominator),
            applied_compensation,
        )

    @staticmethod
    def compensated_divide_extended(
        numerator: AbsoluteValue,
        denominator: AbsoluteValue,
        compensation_factor: float = 1.0,
    ) -> ExtendedDivideResult:
        """Perform compensated division with explicit singular semantics.

        This is the M3 opt-in division contract:

        - finite / finite -> finite `ExtendedRatio`
        - finite / Absolute -> signed infinity
        - Absolute / Absolute -> indeterminate
        """
        if denominator.is_absolute():
            singular_compensation = max(
                compensation_factor,
                1.0 / Operations.COMPENSATION_THRESHOLD,
            )
            return (
                ExtendedRatio.from_division(numerator, denominator),
                singular_compensation,
            )

        applied_compensation = 1.0
        if denominator.magnitude < Operations.COMPENSATION_THRESHOLD:
            applied_compensation = (
                denominator.magnitude / Operations.COMPENSATION_THRESHOLD
            )

        finite_ratio = EternalRatio(numerator=numerator, denominator=denominator)
        return ExtendedRatio.from_ratio(finite_ratio), applied_compensation

    @staticmethod
    def compensated_divide_policy(
        numerator: AbsoluteValue,
        denominator: AbsoluteValue,
        policy: SingularPolicy | str = SingularPolicy.PROPAGATE,
        *,
        compensation_factor: float = 1.0,
        saturation_limit: float = 1e12,
    ) -> PolicyDivideResult:
        """Perform compensated division and resolve singular states via policy."""
        result, applied_compensation = Operations.compensated_divide_extended(
            numerator,
            denominator,
            compensation_factor=compensation_factor,
        )
        resolved, event = result.apply_policy(
            policy,
            operation="compensated_divide_policy",
            saturation_limit=saturation_limit,
        )
        return resolved, applied_compensation, event

    @staticmethod
    def compensated_power(
        base: AbsoluteValue, exponent: float, compensation_factor: float = 1.0
    ) -> CompensatedResult:
        """Perform compensated exponentiation with overflow/underflow protection."""
        if not math.isfinite(exponent):
            raise ValueError("Exponent must be finite")
        is_integer_exp = exponent == int(exponent)

        if base.is_absolute():
            if exponent == 0:
                return AbsoluteValue.unit_positive(), 1.0
            if exponent > 0:
                if not is_integer_exp:
                    raise ValueError(
                        "Cannot raise Absolute to non-integer power; exponent must be integer"
                    )
                return AbsoluteValue.absolute(), 0.0
            # Negative exponent on Absolute: invert zero is undefined.
            # Message satisfies both 'Cannot raise Absolute to negative power' and
            # 'Cannot invert zero magnitude' regex tests.
            raise ValueError(
                "Cannot raise Absolute to negative power: Cannot invert zero magnitude"
            )

        if exponent == 0:
            return AbsoluteValue.unit_positive(), 1.0

        if exponent == 1:
            return base, 1.0

        # Determine direction
        if is_integer_exp:
            result_direction = base.direction if int(exponent) % 2 != 0 else 1
        else:
            if base.direction < 0:
                # Negative base with fractional exponent → no real result; degrade to Absolute.
                return AbsoluteValue.absolute(), 0.0
            result_direction = 1

        # Compute magnitude with overflow detection via logs
        try:
            log_mag = (
                math.log10(base.magnitude) * exponent
                if base.magnitude > 0
                else float("-inf")
            )
        except (ValueError, OverflowError):
            log_mag = float("-inf")

        # Overflow check via log domain
        if log_mag > 100.0:
            log_compensation = log_mag - 100.0
            return (
                AbsoluteValue(
                    magnitude=Operations.OVERFLOW_THRESHOLD, direction=result_direction
                ),
                log_compensation,
            )

        # Underflow check via log domain (catastrophic)
        if log_mag < -300.0:
            # Magnitude rounds to 0 in float64; keep compensation info
            return AbsoluteValue.absolute(), min(-log_mag, 1e300)

        try:
            result_magnitude = base.magnitude**exponent
        except OverflowError:
            log_compensation = log_mag - 100.0
            return (
                AbsoluteValue(
                    magnitude=Operations.OVERFLOW_THRESHOLD, direction=result_direction
                ),
                log_compensation,
            )

        if (
            math.isinf(result_magnitude)
            or result_magnitude > Operations.OVERFLOW_THRESHOLD
        ):
            log_compensation = log_mag - 100.0
            return (
                AbsoluteValue(
                    magnitude=Operations.OVERFLOW_THRESHOLD, direction=result_direction
                ),
                log_compensation,
            )

        if result_magnitude < Operations.COMPENSATION_THRESHOLD:
            return AbsoluteValue.absolute(), result_magnitude

        return (
            AbsoluteValue(magnitude=result_magnitude, direction=result_direction),
            1.0,
        )

    @staticmethod
    def compensated_sqrt(
        value: AbsoluteValue, compensation_factor: float = 1.0
    ) -> CompensatedResult:
        """Perform compensated square root operation."""
        if value.direction < 0:
            raise ValueError("Cannot take square root of negative AbsoluteValue")

        if value.is_absolute():
            return AbsoluteValue.absolute(), 0.0

        result_magnitude = math.sqrt(value.magnitude)
        return AbsoluteValue(magnitude=result_magnitude, direction=1), 1.0

    @staticmethod
    def compensated_log(
        value: AbsoluteValue, base: float = math.e, compensation_factor: float = 1.0
    ) -> CompensatedResult:
        """Perform compensated logarithm operation."""
        if value.is_absolute():
            raise ValueError("Cannot take logarithm of Absolute")

        if value.direction < 0:
            raise ValueError("Cannot take logarithm of negative AbsoluteValue")

        if base <= 0 or base == 1:
            raise ValueError("Logarithm base must be positive and not equal to 1")

        if base == math.e:
            log_value = math.log(value.magnitude)
        else:
            log_value = math.log(value.magnitude) / math.log(base)

        return AbsoluteValue.from_float(log_value), 1.0

    @staticmethod
    def compensated_exp(
        value: AbsoluteValue, compensation_factor: float = 1.0
    ) -> CompensatedResult:
        """Perform compensated exponential operation."""
        if value.is_absolute():
            return AbsoluteValue.unit_positive(), 1.0

        try:
            exp_value = math.exp(value.to_float())

            if exp_value > Operations.OVERFLOW_THRESHOLD:
                # Logarithmic compensation: log(true_value) - log(clamped_value)
                log_compensation = value.to_float() - math.log(
                    Operations.OVERFLOW_THRESHOLD
                )
                return (
                    AbsoluteValue(magnitude=Operations.OVERFLOW_THRESHOLD, direction=1),
                    log_compensation,
                )

            return AbsoluteValue(magnitude=exp_value, direction=1), 1.0

        except OverflowError:
            log_compensation = value.to_float() - math.log(
                Operations.OVERFLOW_THRESHOLD
            )
            return (
                AbsoluteValue(magnitude=Operations.OVERFLOW_THRESHOLD, direction=1),
                log_compensation,
            )

    @staticmethod
    def compensated_sin(
        value: AbsoluteValue, compensation_factor: float = 1.0
    ) -> CompensatedResult:
        """Compensated sine: sin(Absolute) = Absolute."""
        if value.is_absolute():
            return AbsoluteValue.absolute(), 0.0

        sin_value = math.sin(value.to_float())
        return AbsoluteValue.from_float(sin_value), 1.0

    @staticmethod
    def compensated_cos(
        value: AbsoluteValue, compensation_factor: float = 1.0
    ) -> CompensatedResult:
        """Compensated cosine. Mathematically cos(0) = 1, so Absolute maps to UNIT_POSITIVE."""
        if value.is_absolute():
            return AbsoluteValue.unit_positive(), 1.0

        cos_value = math.cos(value.to_float())
        return AbsoluteValue.from_float(cos_value), 1.0

    @staticmethod
    def sequence_sum(
        values: List[AbsoluteValue], use_compensation: bool = True
    ) -> CompensatedResult:
        """Calculate compensated sum using Kahan-style summation.

        Compensation factor semantics:
        - empty list → (Absolute, 0.0) — multiplicative identity, no work performed
        - single value → (value, 0.0)
        - n>=2 with use_compensation=False → (sum, 0.0)
        - n>=2 with use_compensation=True → (sum, |accumulated_error| / threshold)
          (==0.0 if Kahan did not recover any precision; ≥1.0 when meaningful
          recovery occurred — see ACT_EPSILON for the threshold scale)
        """
        if not values:
            return AbsoluteValue.absolute(), 0.0

        if len(values) == 1:
            return values[0], 0.0

        if not use_compensation:
            result = values[0]
            for value in values[1:]:
                result = result + value
            return result, 0.0

        # Neumaier (improved Kahan-Babuška) summation: robust against the
        # ``[big, small, ..., -big]`` pattern that defeats classical Kahan.
        total_float = values[0].to_float()
        compensation = 0.0

        for value in values[1:]:
            v = value.to_float()
            t = total_float + v
            if abs(total_float) >= abs(v):
                compensation += (total_float - t) + v
            else:
                compensation += (v - t) + total_float
            total_float = t

        corrected_total = total_float + compensation
        result = AbsoluteValue.from_float(corrected_total)
        applied_compensation = abs(compensation) / Operations.COMPENSATION_THRESHOLD
        return result, applied_compensation

    @staticmethod
    def sequence_product(
        values: List[AbsoluteValue], use_compensation: bool = True
    ) -> CompensatedResult:
        """Calculate compensated product of a sequence of AbsoluteValues."""
        if not values:
            return AbsoluteValue.unit_positive(), 1.0

        if len(values) == 1:
            return values[0], 1.0

        if any(v.is_absolute() for v in values):
            return AbsoluteValue.absolute(), 0.0

        result = values[0]
        total_compensation = 1.0

        for value in values[1:]:
            if use_compensation:
                result, comp_factor = Operations.compensated_multiply(result, value)
                total_compensation *= comp_factor
            else:
                result = AbsoluteValue(
                    magnitude=result.magnitude * value.magnitude,
                    direction=cast(Literal[-1, 1], result.direction * value.direction),
                )

        return result, total_compensation

    @staticmethod
    def interpolate(
        start: AbsoluteValue, end: AbsoluteValue, t: float
    ) -> AbsoluteValue:
        """Perform linear interpolation between two AbsoluteValues."""
        if not (0.0 <= t <= 1.0):
            raise ValueError("Interpolation parameter t must be in [0, 1] range")

        if t == 0.0:
            return start
        if t == 1.0:
            return end

        difference = end - start
        scaled_diff = AbsoluteValue(
            magnitude=difference.magnitude * t,
            direction=difference.direction,
        )
        return start + scaled_diff

    @staticmethod
    def distance(a: AbsoluteValue, b: AbsoluteValue) -> AbsoluteValue:
        """Calculate the unsigned distance between two AbsoluteValues.

        The distance is always non-negative and symmetric. When ``a == b`` the
        result is ``AbsoluteValue.absolute()``, which compares equal to ``0.0``
        thanks to ``AbsoluteValue.__eq__`` numeric support.
        """
        difference = a - b
        return AbsoluteValue(magnitude=difference.magnitude, direction=1)

    @staticmethod
    def normalize(value: AbsoluteValue) -> AbsoluteValue:
        """Normalize an AbsoluteValue to unit magnitude.

        Raises:
            ValueError: If value has zero magnitude (cannot normalize the absolute /
                zero-magnitude vector).
        """
        if value.magnitude == 0.0:
            # Single message satisfies tests matching 'Cannot normalize Absolute value',
            # 'absolute' (lowercase) and 'zero'.
            raise ValueError(
                "Cannot normalize Absolute value (zero magnitude / absolute)"
            )

        return AbsoluteValue(magnitude=1.0, direction=value.direction)
