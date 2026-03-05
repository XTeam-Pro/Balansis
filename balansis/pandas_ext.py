# Copyright (c) 2024-2026 Tikhonov Andrey. All rights reserved.
# SPDX-License-Identifier: MIT (non-commercial) | Commercial use: see COMMERCIAL_LICENSE.md
from typing import Any
import numpy as np
from balansis.core.absolute import AbsoluteValue

try:
    import pandas as pd
    from pandas.api.extensions import ExtensionDtype, ExtensionArray, register_extension_dtype

    @register_extension_dtype
    class AbsoluteValueDtype(ExtensionDtype):
        name = "absolute"
        type = AbsoluteValue
        kind = "O"

        @classmethod
        def construct_array_type(cls):
            return AbsoluteArray

    class AbsoluteArray(ExtensionArray):
        def __init__(self, values):
            self._data = np.array(values, dtype=object)

        @property
        def dtype(self):
            return AbsoluteValueDtype()

        def __len__(self):
            return self._data.shape[0]

        def __getitem__(self, idx):
            return self._data[idx]

        def isna(self):
            return np.zeros(len(self), dtype=bool)

        def take(self, indices, allow_fill=False, fill_value=None):
            taken = [self._data[i] if i != -1 else fill_value for i in indices]
            return AbsoluteArray(taken)

        def copy(self):
            return AbsoluteArray(self._data.copy())

        def to_numpy(self, dtype=None, copy=False, na_value=None):
            if dtype is None:
                return self._data.copy() if copy else self._data
            if dtype == float or dtype == np.float64:
                return np.array([v.to_float() for v in self._data], dtype=float)
            return self._data.astype(dtype, copy=copy)

        def astype(self, dtype, copy=True):
            arr = self.to_numpy(dtype=dtype, copy=copy)
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                return AbsoluteArray(list(arr))
            return arr

        @classmethod
        def _from_sequence(cls, scalars, dtype=None, copy=False):
            vals = []
            for s in scalars:
                if isinstance(s, AbsoluteValue):
                    vals.append(s)
                elif isinstance(s, (int, float)):
                    vals.append(AbsoluteValue.from_float(float(s)))
                else:
                    vals.append(s)
            return cls(vals)

        def _formatter(self, boxed=False):
            return lambda v: str(v)

except ImportError:
    AbsoluteValueDtype = None
    AbsoluteArray = None
