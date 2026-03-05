# Copyright (c) 2024-2026 Tikhonov Andrey. All rights reserved.
# SPDX-License-Identifier: MIT (non-commercial) | Commercial use: see COMMERCIAL_LICENSE.md
from functools import wraps
from balansis.core.absolute import AbsoluteValue

def safe_computation(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        conv = []
        for a in args:
            if isinstance(a, AbsoluteValue):
                conv.append(a)
            elif isinstance(a, (int, float)):
                conv.append(AbsoluteValue.from_float(float(a)))
            else:
                conv.append(a)
        res = fn(*conv, **kwargs)
        if isinstance(res, AbsoluteValue):
            try:
                return float(res.to_float())
            except Exception:
                return res
        return res
    return wrapper
