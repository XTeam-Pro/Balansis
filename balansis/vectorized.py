# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.
#
# This file is part of Balansis.
# Balansis is dual-licensed under:
#   1. GNU Affero General Public License v3.0 (AGPLv3) for open-source use.
#   2. A Commercial License for proprietary and corporate use.
#
# See LICENSING.md in the project root for license selection details.
# For commercial licensing: andrew@xteam.pro
from typing import List

import numpy as np

from balansis.core.absolute import AbsoluteValue


def batch_add(a: List[AbsoluteValue], b: List[AbsoluteValue]) -> List[AbsoluteValue]:
    if len(a) != len(b):
        raise ValueError("batch_add requires equal lengths")
    return [x + y for x, y in zip(a, b)]


def batch_mul_scalar(a: List[AbsoluteValue], s: float) -> List[AbsoluteValue]:
    return [x * s for x in a]


def batch_to_float(a: List[AbsoluteValue]) -> np.ndarray:
    return np.array([x.to_float() for x in a], dtype=float)
