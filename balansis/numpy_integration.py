from typing import List, Literal, cast

# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.
#
# This file is part of Balansis.
# Balansis is dual-licensed under:
#   1. GNU Affero General Public License v3.0 (AGPLv3) for open-source use.
#   2. A Commercial License for proprietary and corporate use.
#
# See LICENSING.md in the project root for license selection details.
# For commercial licensing: andrew@xteam.pro
import numpy as np

from balansis.core._eft import dot2 as _dot2
from balansis.core.absolute import AbsoluteValue

absolute_struct_dtype = np.dtype([("magnitude", np.float64), ("direction", np.int8)])

# Thresholds aligned with Operations constants from balansis.core.operations
_OVERFLOW_THRESHOLD: float = 1e100
_UNDERFLOW_THRESHOLD: float = 1e-300


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
        out.append(AbsoluteValue(magnitude=m, direction=cast(Literal[-1, 1], d)))
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
    return ufunc_add(ao, bo)  # type: ignore[no-any-return]


def compensated_array_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise addition with Kahan error compensation for near-cancellation.

    Applies the Kahan two-sum correction term so that catastrophic cancellation
    (when |a| ≈ |b| with opposite signs) contributes its residual back into the
    result rather than being silently discarded.

    Args:
        a: First operand array, any shape, will be cast to float64.
        b: Second operand array, same shape as ``a``.

    Returns:
        Compensated sum array of the same shape, dtype float64.
    """
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    result = a64 + b64
    # Kahan two-sum correction: recover the rounding error lost in result = a + b
    comp = (a64 - result) + b64
    return result + comp


def compensated_array_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise multiplication with overflow protection via log-space fallback.

    For elements where the standard product exceeds ``_OVERFLOW_THRESHOLD``,
    the result is recomputed in log-space to avoid IEEE 754 infinity.  All
    other elements use the standard double-precision product.

    Args:
        a: First operand array, any shape, cast to float64.
        b: Second operand array, same shape as ``a``.

    Returns:
        Product array of the same shape, dtype float64.
    """
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    result = a64 * b64
    # Log-space fallback — computed unconditionally so that numpy ufuncs run
    # over the full array; np.where selects it only for overflow positions.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sign = np.sign(a64) * np.sign(b64)
        log_product = np.log(np.abs(a64) + _UNDERFLOW_THRESHOLD) + np.log(
            np.abs(b64) + _UNDERFLOW_THRESHOLD
        )
        log_compensated = sign * np.exp(log_product)
    overflow = np.abs(result) > _OVERFLOW_THRESHOLD
    return np.where(overflow, log_compensated, result)


def compensated_dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Exact finite-input dot product, rounded once to binary64.

    Inputs are cast to float64 and flattened; flattened lengths must match.
    Finite products accumulate exactly using the optional C kernel or integer
    Python reference. Individual products may overflow or underflow float64
    without losing their contribution. Final rounded overflow raises
    OverflowError. Nonfinite inputs retain NumPy propagation via ``dot2``.
    """
    return _dot2(a, b)


def compensated_outer_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Outer product with log-space overflow protection for each element.

    Equivalent to ``np.outer(a, b)`` but uses the same log-space fallback as
    :func:`compensated_array_multiply` to prevent overflow when individual
    elements of ``a`` or ``b`` are very large.

    Args:
        a: 1-D vector of shape (M,), cast to float64.
        b: 1-D vector of shape (N,), cast to float64.

    Returns:
        2-D array of shape (M, N), dtype float64.
    """
    a64 = np.asarray(a, dtype=np.float64).ravel()
    b64 = np.asarray(b, dtype=np.float64).ravel()
    result = np.outer(a64, b64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sign = np.sign(a64)[:, None] * np.sign(b64)[None, :]
        log_product = np.log(np.abs(a64)[:, None] + _UNDERFLOW_THRESHOLD) + np.log(
            np.abs(b64)[None, :] + _UNDERFLOW_THRESHOLD
        )
        log_compensated = sign * np.exp(log_product)
    overflow = np.abs(result) > _OVERFLOW_THRESHOLD
    return np.where(overflow, log_compensated, result)


def compensated_softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax with max-shift stability trick and Kahan denominator summation.

    Combines two standard numerical techniques:
    - Subtract ``max(logits)`` before ``exp`` so the largest exponent is 0,
      preventing overflow for any finite input.
    - Kahan summation for the denominator reduces accumulated rounding error
      when summing many small exp values (common in large-vocabulary settings).

    Args:
        logits: Input score array of any shape, cast to float64.

    Returns:
        Probability array of the same shape, dtype float64, summing to 1.
    """
    logits64 = np.asarray(logits, dtype=np.float64)
    shifted = logits64 - np.max(logits64)
    exp_vals = np.exp(shifted)
    s = np.float64(0.0)
    c = np.float64(0.0)
    for v in exp_vals.ravel():
        y = np.float64(v) - c
        t = s + y
        c = (t - s) - y
        s = t
    # After max-shift, at least one exp_val equals 1.0, so s >= 1 always holds.
    return np.asarray(exp_vals / s, dtype=np.float64)
