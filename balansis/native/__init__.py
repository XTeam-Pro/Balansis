from __future__ import annotations

# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.
#
# This file is part of Balansis.
# Balansis is dual-licensed under:
#   1. GNU Affero General Public License v3.0 (AGPLv3) for open-source use.
#   2. A Commercial License for proprietary and corporate use.
#
# See LICENSING.md in the project root for license selection details.
# For commercial licensing: andrew@xteam.pro
import importlib
from typing import Any, Callable, Iterable, Iterator, Literal, cast

from balansis.core.absolute import AbsoluteValue


def available() -> bool:
    try:
        mod = importlib.import_module("balansis_native")
        return hasattr(mod, "add_absolute")
    except Exception:
        return False


def add_absolute(a: AbsoluteValue, b: AbsoluteValue) -> AbsoluteValue:
    if available():
        mod = importlib.import_module("balansis_native")
        res = mod.add_absolute(a.magnitude, a.direction, b.magnitude, b.direction)
        return AbsoluteValue(magnitude=res[0], direction=res[1])
    return a + b
