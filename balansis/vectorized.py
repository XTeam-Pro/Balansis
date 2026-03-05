# Copyright (c) 2024-2026 Tikhonov Andrey. All rights reserved.
# SPDX-License-Identifier: MIT (non-commercial) | Commercial use: see COMMERCIAL_LICENSE.md
from typing import List
import numpy as np
from balansis.core.absolute import AbsoluteValue

def batch_add(a: List[AbsoluteValue], b: List[AbsoluteValue]) -> List[AbsoluteValue]:
    return [x + y for x, y in zip(a, b)]

def batch_mul_scalar(a: List[AbsoluteValue], s: float) -> List[AbsoluteValue]:
    return [x * s for x in a]

def batch_to_float(a: List[AbsoluteValue]) -> np.ndarray:
    return np.array([x.to_float() for x in a], dtype=float)
