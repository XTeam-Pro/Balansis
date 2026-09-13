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
from typing import TYPE_CHECKING, Any, Callable, Type

import numpy as np

from balansis.core.absolute import AbsoluteValue

try:
    import pandas as pd
    from pandas.api.extensions import (
        ExtensionArray,
        ExtensionDtype,
        register_extension_dtype,
    )

    @register_extension_dtype
    class AbsoluteValueDtype(ExtensionDtype):
        name = "absolute"
        type = AbsoluteValue
        kind = "O"

        @classmethod
        def construct_array_type(cls) -> Type[AbsoluteArray]:
            return AbsoluteArray

    class AbsoluteArray(ExtensionArray):
        def __init__(self, values: Any) -> None:
            self._data = np.array(values, dtype=object)

        @property
        def dtype(self) -> AbsoluteValueDtype:
            return AbsoluteValueDtype()

        def __len__(self) -> int:
            return self._data.shape[0]

        @property
        def nbytes(self) -> int:
            return self._data.nbytes

        @classmethod
        def _concat_same_type(cls, to_concat: list[AbsoluteArray]) -> AbsoluteArray:
            return cls(np.concatenate([array._data for array in to_concat]))

        def __getitem__(self, idx: Any) -> Any:
            result = self._data[idx]
            return AbsoluteArray(result) if isinstance(result, np.ndarray) else result

        def isna(self) -> np.ndarray:
            return np.asarray(pd.isna(self._data), dtype=bool)

        def take(
            self, indices: Any, allow_fill: bool = False, fill_value: Any = None
        ) -> AbsoluteArray:
            from pandas.api.extensions import take

            return AbsoluteArray(
                take(self._data, indices, allow_fill=allow_fill, fill_value=fill_value)
            )

        def copy(self) -> AbsoluteArray:
            return AbsoluteArray(self._data.copy())

        def to_numpy(
            self, dtype: Any = None, copy: bool = False, na_value: Any = None
        ) -> np.ndarray:
            if dtype is None:
                return self._data.copy() if copy else self._data
            if dtype == float or dtype == np.float64:
                return np.array(
                    [
                        v.to_float() if isinstance(v, AbsoluteValue) else np.nan
                        for v in self._data
                    ],
                    dtype=float,
                )
            return self._data.astype(dtype, copy=copy)

        def astype(self, dtype: Any, copy: bool = True) -> Any:
            arr = self.to_numpy(dtype=dtype, copy=copy)
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                return AbsoluteArray(list(arr))
            return arr

        @classmethod
        def _from_sequence(
            cls, scalars: Any, dtype: Any = None, copy: bool = False
        ) -> AbsoluteArray:
            vals = []
            for s in scalars:
                if isinstance(s, AbsoluteValue):
                    vals.append(s)
                elif isinstance(s, (int, float)):
                    vals.append(AbsoluteValue.from_float(float(s)))
                else:
                    vals.append(s)
            return cls(vals)

        def _formatter(self, boxed: bool = False) -> Callable[[Any], str]:
            return lambda v: str(v)

except ImportError:
    if not TYPE_CHECKING:
        AbsoluteValueDtype = None
        AbsoluteArray = None
