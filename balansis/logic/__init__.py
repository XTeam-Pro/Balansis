# Copyright (c) 2024-2026 Tikhonov Andrey. All rights reserved.
# SPDX-License-Identifier: MIT (non-commercial) | Commercial use: see COMMERCIAL_LICENSE.md
"""Logic and compensation engine for Balansis library.

This module contains the compensation logic that ensures balance and stability:
- Compensator: Core engine for balance calculations and stability verification
"""

from .compensator import Compensator

__all__ = ["Compensator"]