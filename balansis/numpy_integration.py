import math

import numpy as np
from typing import List

from balansis.core.absolute import AbsoluteValue
from balansis.core.operations import Operations

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


def compensated_array_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise ACT-compensated addition of two float arrays.

    Converts elements to AbsoluteValue, performs compensated addition,
    and converts back to float.

    Args:
        a: First input array.
        b: Second input array (must be same shape as a).

    Returns:
        Array of compensated sums.
    """
    flat_a = a.flatten()
    flat_b = b.flatten()
    result = np.empty(flat_a.size, dtype=np.float64)
    for i in range(flat_a.size):
        av_a = AbsoluteValue.from_float(float(flat_a[i]))
        av_b = AbsoluteValue.from_float(float(flat_b[i]))
        res, _ = Operations.compensated_add(av_a, av_b)
        result[i] = res.to_float()
    return result.reshape(a.shape)


def compensated_array_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise ACT-compensated multiplication of two float arrays.

    Args:
        a: First input array.
        b: Second input array (must be same shape as a).

    Returns:
        Array of compensated products.
    """
    flat_a = a.flatten()
    flat_b = b.flatten()
    result = np.empty(flat_a.size, dtype=np.float64)
    for i in range(flat_a.size):
        av_a = AbsoluteValue.from_float(float(flat_a[i]))
        av_b = AbsoluteValue.from_float(float(flat_b[i]))
        res, _ = Operations.compensated_multiply(av_a, av_b)
        result[i] = res.to_float()
    return result.reshape(a.shape)


def compensated_dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """ACT-compensated dot product using Operations.sequence_sum.

    Computes elementwise products via compensated multiplication,
    then sums them using Kahan-compensated sequence_sum.

    Args:
        a: First input vector.
        b: Second input vector (must be same length as a).

    Returns:
        Compensated dot product as float.
    """
    flat_a = a.flatten()
    flat_b = b.flatten()
    products = []
    for i in range(flat_a.size):
        av_a = AbsoluteValue.from_float(float(flat_a[i]))
        av_b = AbsoluteValue.from_float(float(flat_b[i]))
        prod, _ = Operations.compensated_multiply(av_a, av_b)
        products.append(prod)
    result, _ = Operations.sequence_sum(products)
    return result.to_float()


def compensated_outer_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ACT-compensated outer product.

    Args:
        a: First input vector.
        b: Second input vector.

    Returns:
        2D array where result[i,j] = ACT-compensated a[i] * b[j].
    """
    flat_a = a.flatten()
    flat_b = b.flatten()
    result = np.empty((flat_a.size, flat_b.size), dtype=np.float64)
    for i in range(flat_a.size):
        av_a = AbsoluteValue.from_float(float(flat_a[i]))
        for j in range(flat_b.size):
            av_b = AbsoluteValue.from_float(float(flat_b[j]))
            prod, _ = Operations.compensated_multiply(av_a, av_b)
            result[i, j] = prod.to_float()
    return result


def compensated_softmax(logits: np.ndarray) -> np.ndarray:
    """ACT-compensated softmax for stable probability computation.

    Applies max-subtraction for numerical stability, then computes
    exp and sum using ACT-compensated operations.

    Args:
        logits: Input logit array.

    Returns:
        Probability array that sums to 1.
    """
    flat = logits.flatten()
    max_logit = float(np.max(flat))

    # Compute exp(logit - max) using ACT-compensated exp
    exp_values = []
    for i in range(flat.size):
        shifted = AbsoluteValue.from_float(float(flat[i]) - max_logit)
        exp_val, _ = Operations.compensated_exp(shifted)
        exp_values.append(exp_val)

    # Compensated sum of exp values
    exp_sum, _ = Operations.sequence_sum(exp_values)

    # Divide each by sum using compensated division
    result = np.empty(flat.size, dtype=np.float64)
    for i in range(flat.size):
        if exp_sum.magnitude > 0:
            ratio, _ = Operations.compensated_divide(exp_values[i], exp_sum)
            result[i] = ratio.numerical_value()
        else:
            result[i] = 1.0 / flat.size

    return result.reshape(logits.shape)
