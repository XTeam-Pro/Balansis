# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.
#
# This file is part of Balansis.
# Balansis is dual-licensed under:
#   1. GNU Affero General Public License v3.0 (AGPLv3) for open-source use.
#   2. A Commercial License for proprietary and corporate use.
#
# See LICENSING.md in the project root for license selection details.
# For commercial licensing: andrew@xteam.pro
"""Core mathematical types and operations for Balansis library.

This module contains the fundamental mathematical constructs of Absolute Compensation Theory:
- AbsoluteValue: Core value type with magnitude and direction
- EternalRatio: Structural ratios between AbsoluteValues
- Operations: Basic mathematical operations following ACT principles
"""

from .absolute import AbsoluteValue
from .eternity import (
    EternalRatio,
    ExtendedRatio,
    SingularArithmeticEvent,
    SingularPolicy,
)
from .operations import Operations

__all__ = [
    "AbsoluteValue",
    "EternalRatio",
    "ExtendedRatio",
    "SingularPolicy",
    "SingularArithmeticEvent",
    "Operations",
]
