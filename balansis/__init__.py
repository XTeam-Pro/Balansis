"""Balansis: Python mathematical library implementing Absolute Compensation Theory (ACT).

This library provides a novel framework replacing traditional zero and infinity
with Absolute and Eternity concepts for enhanced computational stability.

Core Components:
    - AbsoluteValue: Values with magnitude and direction
    - EternalRatio: Structural ratios between AbsoluteValues
    - Operations: Compensated arithmetic operations
    - Compensator: Balance and stability calculations
    - AbsoluteGroup: Group theory for Absolute values
    - EternityField: Field operations for eternal ratios

Example:
    >>> from balansis import AbsoluteValue, EternalRatio
    >>> a = AbsoluteValue(magnitude=5.0, direction=1)
    >>> b = AbsoluteValue(magnitude=3.0, direction=-1)
    >>> result = a + b  # Compensated addition
    >>> ratio = EternalRatio(numerator=a, denominator=b)
"""

from typing import Union

from balansis.core.absolute import AbsoluteValue
from balansis.core.eternity import EternalRatio
from balansis.core.operations import Operations
from balansis.logic.compensator import Compensator
from balansis.algebra.absolute_group import AbsoluteGroup
from balansis.algebra.eternity_field import EternityField
from balansis.sets.eternal_set import EternalSet
from balansis.sets.resolver import global_compensate
from balansis.sets.generators import harmonic_generator, grandis_generator
# from balansis.utils.plot import PlotUtils  # Temporarily disabled

__version__ = "0.1.0"
__author__ = "Balansis Team"
__email__ = "team@balansis.org"

# ACT Constants
ABSOLUTE = AbsoluteValue(magnitude=0.0, direction=1)
UNIT_POSITIVE = AbsoluteValue(magnitude=1.0, direction=1)
UNIT_NEGATIVE = AbsoluteValue(magnitude=1.0, direction=-1)

# Mathematical limits and tolerances
DEFAULT_TOLERANCE = 1e-10
STABILITY_THRESHOLD = 1e-8
MAX_MAGNITUDE = 1e308
MIN_MAGNITUDE = 1e-308

# ACT-specific constants
ACT_EPSILON = 1e-15
ACT_STABILITY_THRESHOLD = 1e-12
ACT_ABSOLUTE_THRESHOLD = 1e-20
ACT_COMPENSATION_FACTOR = 0.1

__all__ = [
    "AbsoluteValue",
    "EternalRatio",
    "Operations",
    "Compensator",
    "AbsoluteGroup",
    "EternityField",
    "EternalSet",
    "global_compensate",
    "harmonic_generator",
    "grandis_generator",
    # "PlotUtils",  # Temporarily disabled
    "ABSOLUTE",
    "UNIT_POSITIVE",
    "UNIT_NEGATIVE",
    "DEFAULT_TOLERANCE",
    "STABILITY_THRESHOLD",
    "MAX_MAGNITUDE",
    "MIN_MAGNITUDE",
    "ACT_EPSILON",
    "ACT_STABILITY_THRESHOLD",
    "ACT_ABSOLUTE_THRESHOLD",
    "ACT_COMPENSATION_FACTOR",
    "B",
]


def B(value: Union[int, float, str, AbsoluteValue]) -> AbsoluteValue:
    """Convert a numeric value to an AbsoluteValue.

    Convenience constructor that accepts int, float, str, or AbsoluteValue
    and returns an AbsoluteValue instance via AbsoluteValue.from_float().

    Args:
        value: The value to convert. Strings are first parsed as floats.

    Returns:
        AbsoluteValue representing the input value.

    Raises:
        TypeError: If value is not int, float, str, or AbsoluteValue.
        ValueError: If string cannot be parsed as float.
    """
    if isinstance(value, (int, float)):
        return AbsoluteValue.from_float(float(value))
    if isinstance(value, str):
        v = float(value)
        return AbsoluteValue.from_float(v)
    if isinstance(value, AbsoluteValue):
        return value
    raise TypeError("unsupported type for AbsoluteValue alias")
