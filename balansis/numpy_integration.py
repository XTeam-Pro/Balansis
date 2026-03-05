# Copyright (c) 2024-2026 Tikhonov Andrey. All rights reserved.
# SPDX-License-Identifier: MIT (non-commercial) | Commercial use: see COMMERCIAL_LICENSE.md
import numpy as np
from typing import List
from balansis.core.absolute import AbsoluteValue

absolute_struct_dtype = np.dtype([("magnitude", np.float64), ("direction", np.int8)])

def to_numpy(values: List[AbsoluteValue]) -> np.ndarray:
    arr = np.empty(len(values), dtype=absolute_struct_dtype)
    for i, v in enumerate(values):
        arr[i] = (v.magnitude, v.direction)
    return arr

def from_numpy(arr: np.ndarray) -> List[AbsoluteValue]:
    out: List[AbsoluteValue] = []
    for i in range(arr.shape[0]):
        m = float(arr["magnitude"][i])
        d = int(arr["direction"][i])
        out.append(AbsoluteValue(magnitude=m, direction=d))
    return out

ufunc_add = np.frompyfunc(lambda a, b: a + b, 2, 1)
ufunc_sub = np.frompyfunc(lambda a, b: a - b, 2, 1)
ufunc_mul_scalar = np.frompyfunc(lambda a, s: a * float(s), 2, 1)
ufunc_log = np.frompyfunc(lambda a: a.log(), 1, 1)
ufunc_exp = np.frompyfunc(lambda a: a.exp(), 1, 1)
ufunc_sin = np.frompyfunc(lambda a: a.sin(), 1, 1)
ufunc_cos = np.frompyfunc(lambda a: a.cos(), 1, 1)
ufunc_tan = np.frompyfunc(lambda a: a.tan(), 1, 1)

def add_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ao = a.astype(object)
    bo = b.astype(object)
    return ufunc_add(ao, bo)
