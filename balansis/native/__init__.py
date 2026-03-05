# Copyright (c) 2024-2026 Tikhonov Andrey. All rights reserved.
# SPDX-License-Identifier: MIT (non-commercial) | Commercial use: see COMMERCIAL_LICENSE.md
import importlib
from balansis.core.absolute import AbsoluteValue

def available():
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
