# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.
#
# This file is part of Balansis.
# Balansis is dual-licensed under:
#   1. GNU Affero General Public License v3.0 (AGPLv3) for open-source use.
#   2. A Commercial License for proprietary and corporate use.
#
# See LICENSING.md in the project root for license selection details.
# For commercial licensing: andrew@xteam.pro
"""Logic and compensation engine for Balansis library.

This module contains the compensation logic that ensures balance and stability:
- Compensator: Core engine for balance calculations and stability verification
"""

from .compensator import Compensator

__all__ = ["Compensator"]
