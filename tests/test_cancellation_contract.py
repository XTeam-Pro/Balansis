"""Cancellation must respect the represented operands, including their order."""

import math
from fractions import Fraction

import pytest

from balansis import AbsoluteValue, Operations


@pytest.mark.parametrize("magnitude", [1.0, 9e15, 1e16, 1e20, 1e100, 1e300])
@pytest.mark.parametrize("direction", [-1, 1])
def test_exact_opposites_cancel_at_every_magnitude(magnitude, direction):
    left = AbsoluteValue.from_float(direction * magnitude)
    right = AbsoluteValue.from_float(-direction * magnitude)
    result, compensation = Operations.compensated_add(left, right)

    assert result.is_absolute()
    assert result.to_float() == float(
        Fraction(left.to_float()) + Fraction(right.to_float())
    )
    assert math.isfinite(compensation)


@pytest.mark.parametrize("magnitude", [1e16, 1e20, 1e100, 1e300])
def test_distinct_neighbors_preserve_the_represented_difference(magnitude):
    smaller = math.nextafter(magnitude, 0.0)
    exact = float(Fraction(magnitude) - Fraction(smaller))
    left = AbsoluteValue.from_float(magnitude)
    right = AbsoluteValue.from_float(-smaller)

    forward, forward_compensation = Operations.compensated_add(left, right)
    reverse, reverse_compensation = Operations.compensated_add(right, left)

    assert forward.to_float() == reverse.to_float() == exact
    assert forward_compensation == reverse_compensation


def test_rounding_before_construction_cannot_be_recovered_by_addition():
    rounded = 1e16 + 1.0
    assert rounded == 1e16
    result, _ = Operations.compensated_add(
        AbsoluteValue.from_float(rounded), AbsoluteValue.from_float(-1e16)
    )
    assert result.is_absolute()
