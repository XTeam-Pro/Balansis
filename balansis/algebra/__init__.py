# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.
#
# This file is part of Balansis.
# Balansis is dual-licensed under:
#   1. GNU Affero General Public License v3.0 (AGPLv3) for open-source use.
#   2. A Commercial License for proprietary and corporate use.
#
# See LICENSING.md in the project root for license selection details.
# For commercial licensing: andrew@xteam.pro
"""Algebraic structures for Balansis library.

This module implements advanced algebraic structures based on ACT principles:
- AbsoluteGroup: Group theory operations for Absolute values
- EternityField: Field operations for eternal ratios
"""

from .absolute_group import AbsoluteGroup
from .eternity_field import EternityField

__all__ = ["AbsoluteGroup", "EternityField"]
