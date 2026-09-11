# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.
#
# This file is part of Balansis.
# Balansis is dual-licensed under:
#   1. GNU Affero General Public License v3.0 (AGPLv3) for open-source use.
#   2. A Commercial License for proprietary and corporate use.
#
# See LICENSING.md in the project root for license selection details.
# For commercial licensing: andrew@xteam.pro
"""Compensated arithmetic, numerical arrays, algebra and singular ratios."""

from balansis.algebra.absolute_group import AbsoluteGroup
from balansis.algebra.eternity_field import EternityField
from balansis.array import dot_array, gram_pair, sum_array
from balansis.compat import CompensatedMatMul, CompensatedSum, StableSoftmax
from balansis.core.absolute import AbsoluteValue
from balansis.core.eternity import (
    EternalRatio,
    ExtendedRatio,
    SingularArithmeticEvent,
    SingularPolicy,
)
from balansis.core.operations import Operations
from balansis.logic.compensator import Compensator
from balansis.sets.eternal_set import EternalSet
from balansis.sets.generators import grandis_generator, harmonic_generator
from balansis.sets.resolver import global_compensate

__version__ = "1.2.0"
__author__ = "Andrey Tikhonov (XTeam-Pro)"
__email__ = "andrew@xteam.pro"
__license__ = (
    "AGPL-3.0 / Commercial — see LICENSE, LICENSING.md, and COMMERCIAL_LICENSE.md"
)

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
    "ExtendedRatio",
    "SingularPolicy",
    "SingularArithmeticEvent",
    "Operations",
    "sum_array",
    "dot_array",
    "gram_pair",
    "Compensator",
    "AbsoluteGroup",
    "EternityField",
    "EternalSet",
    "global_compensate",
    "harmonic_generator",
    "grandis_generator",
    "CompensatedSum",
    "StableSoftmax",
    "CompensatedMatMul",
    "B",
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
]


def B(value: int | float | str | AbsoluteValue) -> AbsoluteValue:
    if isinstance(value, (int, float)):
        return AbsoluteValue.from_float(float(value))
    if isinstance(value, str):
        v = float(value)
        return AbsoluteValue.from_float(v)
    if isinstance(value, AbsoluteValue):
        return value
    raise TypeError("unsupported type for AbsoluteValue alias")
