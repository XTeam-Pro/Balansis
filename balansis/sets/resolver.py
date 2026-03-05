# Copyright (c) 2024-2026 Tikhonov Andrey. All rights reserved.
# SPDX-License-Identifier: MIT (non-commercial) | Commercial use: see COMMERCIAL_LICENSE.md
import itertools
import math
from typing import Iterator
from balansis.core.absolute import AbsoluteValue
from balansis.sets.eternal_set import EternalSet

def global_compensate(set_a: EternalSet, set_b: EternalSet) -> EternalSet:
    zero = AbsoluteValue.absolute()
    def generator() -> Iterator[AbsoluteValue]:
        for a, b in itertools.zip_longest(set_a, set_b, fillvalue=zero):
            r = a + b
            if hasattr(r, "is_absolute"):
                if r.is_absolute():
                    if set_a.is_infinite or set_b.is_infinite:
                        yield zero
                    else:
                        continue
            else:
                if math.isclose(r.magnitude, 0.0, rel_tol=1e-12, abs_tol=1e-12):
                    if set_a.is_infinite or set_b.is_infinite:
                        yield zero
                    else:
                        continue
            yield r
    return EternalSet(generator(), is_infinite=(set_a.is_infinite or set_b.is_infinite), rule_name="global_compensate")

def verify_zero_sum(result_set: EternalSet, threshold: int = 1000):
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

def stream_compensate(iter1, iter2, limit: int | None = None):
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

def convergence_detector(result_iter, window: int = 100, tol: float = 1e-12) -> bool:
    buf = []
    for x in result_iter:
        buf.append(x)
        if len(buf) > window:
            buf.pop(0)
        if len(buf) == window:
            if all(v.is_absolute() or abs(v.magnitude) <= tol for v in buf):
                return True
    return False
