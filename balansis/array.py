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


def native_dot_available() -> bool:
    """Whether the installed optional extension supports exact dot products."""
    return native_available() and callable(getattr(_kernels, "exact_dot", None))


def native_gram_available() -> bool:
    """Whether the extension supports the fused three-product kernel."""
    return native_available() and callable(getattr(_kernels, "gram_pair", None))


def gram_pair(
    a: ArrayLike,
    b: ArrayLike,
    *,
    backend: Literal["auto", "python", "native"] = "auto",
) -> tuple[float, float, float]:
    """Return exact (dot(a,a), dot(b,b), dot(a,b)), each rounded once.

    Finite real 1-D inputs must have equal lengths. Any rounded entry overflowing
    binary64 raises OverflowError. ``native`` requires the fused kernel; ``auto``
    also supports older extensions via three exact dots, or the Python reference.
    Contiguous native float64 buffers need no input copy.
    """
    if backend not in ("auto", "python", "native"):
        raise ValueError("backend must be 'auto', 'python', or 'native'")
    use_native = backend != "python" and native_gram_available()
    if backend == "native" and not use_native:
        raise RuntimeError(
            "native Gram backend unavailable; install ./native from source"
        )
    left, right = np.asarray(a), np.asarray(b)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("gram_pair requires one-dimensional arrays")
    if left.size != right.size:
        raise ValueError("gram_pair requires equal lengths")
    if left.dtype.kind not in "biuf" or right.dtype.kind not in "biuf":
        raise TypeError("gram_pair requires real numeric values")
    left = np.ascontiguousarray(left, dtype=np.float64)
    right = np.ascontiguousarray(right, dtype=np.float64)
    if use_native and _kernels is not None:
        aa, bb, ab = _kernels.gram_pair(left, right)
        return float(aa), float(bb), float(ab)
    # Prevalidate both vectors, including with older kernels, so a nonfinite
    # input takes precedence over a norm overflow in the fallback as well.
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("gram_pair requires finite values")
    if backend != "python" and native_dot_available() and _kernels is not None:
        return (
            float(_kernels.exact_dot(left, left)),
            float(_kernels.exact_dot(right, right)),
            float(_kernels.exact_dot(left, right)),
        )
    return _python_dot(left, left), _python_dot(right, right), _python_dot(left, right)


def _python_dot(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Integer reference: all finite binary64 products are multiples of 2^-2148."""
    total = 0
    for left, right in zip(a, b):
        x, y = float(left), float(right)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("dot_array requires finite values")
        nx, dx = x.as_integer_ratio()
        ny, dy = y.as_integer_ratio()
        shift = 2148 - (dx.bit_length() - 1) - (dy.bit_length() - 1)
        total += (nx * ny) << shift
    # Integer true division rounds once, preserving negative underflow to -0.0.
    return total / (1 << 2148)


def dot_array(
    a: ArrayLike,
    b: ArrayLike,
    *,
    backend: Literal["auto", "python", "native"] = "auto",
) -> float:
    """Exact sum of represented float64 products, rounded once to nearest-even.

    Real 1-D inputs must have equal lengths. Conversion to float64 may round
    the inputs first. NaN/infinity raise ValueError; final rounded overflow
    raises OverflowError. Products may exceed float64 range and cancel later.
    The optional C kernel and integer Python reference use the same contract.
    Contiguous native float64 inputs need no input copy. Exact zero is +0.0;
    negative nonzero values rounding to zero produce -0.0.
    """
    if backend not in ("auto", "python", "native"):
        raise ValueError("backend must be 'auto', 'python', or 'native'")
    use_native = backend != "python" and native_dot_available()
    if backend == "native" and not use_native:
        raise RuntimeError(
            "native dot backend unavailable; install ./native from source"
        )
    left, right = np.asarray(a), np.asarray(b)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("dot_array requires one-dimensional arrays")
    if left.size != right.size:
        raise ValueError("dot_array requires equal lengths")
    if left.dtype.kind not in "biuf" or right.dtype.kind not in "biuf":
        raise TypeError("dot_array requires real numeric values")
    left = np.ascontiguousarray(left, dtype=np.float64)
    right = np.ascontiguousarray(right, dtype=np.float64)
    if use_native and _kernels is not None:
        return float(_kernels.exact_dot(left, right))
    return _python_dot(left, right)


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
