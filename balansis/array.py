"""Batch compensated arithmetic on real, one-dimensional arrays."""

import importlib
import math
from types import ModuleType
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from balansis.core.absolute import AbsoluteValue
from balansis.core.operations import Operations

_kernels: ModuleType | None
try:
    _kernels = importlib.import_module("_balansis_kernels")
except ModuleNotFoundError as exc:
    if exc.name != "_balansis_kernels":
        raise
    _kernels = None


def native_available() -> bool:
    """Whether the optional C extension implements the expected kernel API."""
    return _kernels is not None and getattr(_kernels, "API_VERSION", None) == 1


def _python_sum(values: NDArray[np.float64]) -> tuple[float, float]:
    """Reference kernel with the same operation order and errors as C."""
    if values.size == 0:
        return 0.0, 0.0
    total = float(values[0])
    if not math.isfinite(total):
        raise ValueError("sum_array requires finite values")
    correction = 0.0
    for item in values[1:]:
        value = float(item)
        if not math.isfinite(value):
            raise ValueError("sum_array requires finite values")
        next_total = total + value
        if not math.isfinite(next_total):
            raise OverflowError("compensated sum overflowed float64")
        if abs(total) >= abs(value):
            correction += (total - next_total) + value
        else:
            correction += (value - next_total) + total
        if not math.isfinite(correction):
            raise OverflowError("compensated sum overflowed float64")
        total = next_total
    result = total + correction
    if not math.isfinite(result):
        raise OverflowError("compensated sum overflowed float64")
    return result, correction


def sum_array(
    values: ArrayLike, *, backend: Literal["auto", "python", "native"] = "auto"
) -> tuple[AbsoluteValue, float]:
    """Sum a 1-D real array with sequential Neumaier compensation.

    Returns the same result shape and diagnostic scaling as
    ``Operations.sequence_sum``. The diagnostic is NOT an error bound.
    A contiguous native float64 ndarray needs no input copy; other real numeric
    arrays are converted. NaN/infinity raise ValueError; intermediate or final
    overflow raises OverflowError even if exact arithmetic would cancel later.
    ``native`` requires the optional balansis-kernels extension; ``auto`` uses
    the Python reference when that extension is absent or has a different API.
    No correctly-rounded or arbitrary-precision guarantee is made.
    """
    if backend not in ("auto", "python", "native"):
        raise ValueError("backend must be 'auto', 'python', or 'native'")
    use_native = backend != "python" and native_available()
    if backend == "native" and not use_native:
        raise RuntimeError("native backend unavailable; install ./native from source")
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("sum_array requires a one-dimensional array")
    if array.dtype.kind not in "biuf":
        raise TypeError("sum_array requires real numeric values")
    array = np.ascontiguousarray(array, dtype=np.float64)
    if use_native and _kernels is not None:
        result, correction = _kernels.neumaier_sum(array)
    else:
        result, correction = _python_sum(array)
    return (
        AbsoluteValue.from_float(result),
        abs(correction) / Operations.COMPENSATION_THRESHOLD,
    )
