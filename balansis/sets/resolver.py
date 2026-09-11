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
import itertools
import math
from typing import Iterable, Iterator

from balansis.core.absolute import AbsoluteValue
from balansis.sets.eternal_set import EternalSet


def global_compensate(set_a: EternalSet, set_b: EternalSet) -> EternalSet:
    """Combine corresponding terms; infinite streams retain one zero per pair."""
    zero = AbsoluteValue.absolute()
    infinite = set_a.is_infinite or set_b.is_infinite

    def generator() -> Iterator[AbsoluteValue]:
        for a, b in itertools.zip_longest(set_a, set_b, fillvalue=zero):
            result = a + b
            if not result.is_absolute() or infinite:
                yield result

    return EternalSet(generator(), is_infinite=infinite, rule_name="global_compensate")


def verify_zero_sum(
    result_set: EternalSet, threshold: int = 1000
) -> list[AbsoluteValue]:
    residuals = []
    it = iter(result_set)
    for _ in range(int(threshold)):
        try:
            x = next(it)
        except StopIteration:
            break
        if hasattr(x, "is_absolute"):
            if x.is_absolute():
                continue
        else:
            if math.isclose(x.magnitude, 0.0, rel_tol=1e-12, abs_tol=1e-12):
                continue
        residuals.append(x)
    return residuals


def stream_compensate(
    iter1: Iterable[AbsoluteValue],
    iter2: Iterable[AbsoluteValue],
    limit: int | None = None,
) -> Iterator[AbsoluteValue]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return
    zero = AbsoluteValue.absolute()
    count = 0
    for a, b in itertools.zip_longest(iter1, iter2, fillvalue=zero):
        r = a + b
        if r.is_absolute():
            pass
        else:
            yield r
        count += 1
        if limit is not None and count >= limit:
            break


def convergence_detector(
    result_iter: Iterable[AbsoluteValue], window: int = 100, tol: float = 1e-12
) -> bool:
    if window <= 0 or not math.isfinite(tol) or tol < 0:
        raise ValueError("window must be positive and tol finite and non-negative")
    buf = []
    for x in result_iter:
        buf.append(x)
        if len(buf) > window:
            buf.pop(0)
        if len(buf) == window:
            if all(v.is_absolute() or abs(v.magnitude) <= tol for v in buf):
                return True
    return False
