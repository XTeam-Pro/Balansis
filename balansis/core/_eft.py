# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.
#
# This file is part of Balansis.
# Balansis is dual-licensed under:
#   1. GNU Affero General Public License v3.0 (AGPLv3) for open-source use.
#   2. A Commercial License for proprietary and corporate use.
#
# See LICENSING.md in the project root for license selection details.
# For commercial licensing: andrew@xteam.pro
"""Classical floating-point transforms and exact finite-input dot products.

TwoSum and Dekker TwoProduct are floating-point error transforms whose exactness
requires suitable range conditions; splitting can overflow and product residuals
can underflow. ``dot2`` uses the separate exact integer accumulator through
``dot_array`` rather than relying on those transforms for full-range inputs.
"""

from __future__ import annotations

import math
from typing import Any, Tuple

import numpy as np

# 2^27 + 1 — the Dekker splitting constant for float64 (26-bit halves).
_SPLIT = 134217729.0


def two_sum(a: float, b: float) -> Tuple[float, float]:
    """Return ``(s, e)`` with ``s = fl(a + b)`` and ``a + b == s + e`` exactly."""
    s = a + b
    bb = s - a
    err = (a - (s - bb)) + (b - bb)
    return s, err


def two_product(a: float, b: float) -> Tuple[float, float]:
    """Return ``(p, e)`` with ``p = fl(a * b)`` and ``a * b == p + e`` exactly."""
    p = a * b
    c = _SPLIT * a
    ah = c - (c - a)
    al = a - ah
    d = _SPLIT * b
    bh = d - (d - b)
    bl = b - bh
    err = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, err


def _split_arr(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    c = _SPLIT * v
    hi = c - (c - v)
    lo = v - hi
    return hi, lo


def two_product_arr(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized :func:`two_product`. Returns ``(p, e)`` arrays with ``a*b == p+e``."""
    p = a * b
    ah, al = _split_arr(a)
    bh, bl = _split_arr(b)
    err = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, err


def dot2(a: Any, b: Any) -> float:
    """Flatten inputs and compute an exact finite binary64 dot product.

    Finite values use ``dot_array`` with its native or integer Python backend.
    Nonfinite values retain the legacy NumPy propagation path. Length mismatch
    is rejected instead of broadcasting. Final rounded overflow raises
    OverflowError; overflowing products that cancel are handled exactly.
    """
    from balansis.array import dot_array

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError("dot2 requires equal lengths")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float(np.dot(a, b))
    return dot_array(a, b)


def comp_sum(values: Any) -> float:
    """Correctly rounded sum of a 1-D set of floats (exact summation)."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    if not np.isfinite(arr).all():
        return float(np.sum(arr))
    return math.fsum(arr)
