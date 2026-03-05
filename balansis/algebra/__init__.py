# Copyright (c) 2024-2026 Tikhonov Andrey. All rights reserved.
# SPDX-License-Identifier: MIT (non-commercial) | Commercial use: see COMMERCIAL_LICENSE.md
"""Algebraic structures for Balansis library.

This module implements advanced algebraic structures based on ACT principles:
- AbsoluteGroup: Group theory operations for Absolute values
- EternityField: Field operations for eternal ratios
"""

from .absolute_group import AbsoluteGroup
from .eternity_field import EternityField

__all__ = ["AbsoluteGroup", "EternityField"]