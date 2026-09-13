"""Numerical, buffer-boundary and dispatch contracts of batch summation."""

import array
import math
import struct
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

import balansis.array as batch
from balansis import AbsoluteValue, Operations, sum_array

BACKENDS = ["python", "auto", "native"]


@pytest.fixture(params=BACKENDS)
def backend(request):
    if request.param == "native" and not batch.native_available():
        pytest.skip("optional C extension not installed")
    return request.param


@pytest.mark.parametrize(
    "values",
    [
        [],
        [0.0],
        [-0.0],
        [7.0],
        [-7.0],
        [1e16, 1.0, -1e16],
        [1.0, 1e100, 1.0, -1e100],
        [1e300, -1e300],
        [math.nextafter(1e16, math.inf), -1e16],
        [math.ulp(0.0), math.ulp(0.0)],
        [1.0, math.ulp(0.0), -1.0],
        [np.finfo(float).max, -np.finfo(float).max],
        [1e16, 1.0, -1e16] * 200,
    ],
)
def test_exact_representable_examples(values, backend):
    expected = float(sum((Fraction(float(x)) for x in values), Fraction()))
    result, diagnostic = sum_array(values, backend=backend)
    assert result.to_float() == expected
    legacy = Operations.sequence_sum([AbsoluteValue.from_float(x) for x in values])
    assert result.to_float() == legacy[0].to_float()
    assert diagnostic == legacy[1]


def test_seeded_wide_exponent_parity(backend):
    rng = np.random.default_rng(731)
    for _ in range(30):
        values = np.ldexp(rng.uniform(-1, 1, 256), rng.integers(-900, 900, 256))
        expected = Operations.sequence_sum(
            [AbsoluteValue.from_float(float(x)) for x in values]
        )
        actual = sum_array(values, backend=backend)
        assert actual[0].to_float() == expected[0].to_float()
        assert actual[1] == expected[1]


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
@pytest.mark.parametrize("position", [0, 1, 2])
def test_nonfinite_inputs_rejected(value, position, backend):
    values = [1.0, 2.0, 3.0]
    values[position] = value
    with pytest.raises(ValueError, match="finite"):
        sum_array(values, backend=backend)


@pytest.mark.parametrize(
    "values",
    [
        [np.finfo(float).max, np.finfo(float).max],
        [-np.finfo(float).max, -np.finfo(float).max],
        [np.finfo(float).max, np.finfo(float).max, -np.finfo(float).max],
        [np.finfo(float).max, 2.0**969, 2.0**969, 2.0**969, 2.0**969],
    ],
)
def test_overflow_rejected_including_intermediate_and_final(values, backend):
    with pytest.raises(OverflowError, match="overflowed"):
        sum_array(values, backend=backend)


@pytest.mark.parametrize("values", [3.0, [[1.0, 2.0]], np.zeros((2, 0))])
def test_rank_rejected(values, backend):
    with pytest.raises(ValueError, match="one-dimensional"):
        sum_array(values, backend=backend)


@pytest.mark.parametrize("values", [[1j], ["1.0"], np.array([1], dtype=object)])
def test_nonreal_dtypes_rejected(values, backend):
    with pytest.raises(TypeError, match="real numeric"):
        sum_array(values, backend=backend)


@pytest.mark.parametrize("dtype", [np.int64, np.float32, ">f8", "<f8"])
def test_converted_inputs(dtype, backend):
    result, _ = sum_array(np.array([3, -1, 2], dtype=dtype), backend=backend)
    assert result.to_float() == 4.0


def test_views_readonly_and_unaligned_inputs(backend):
    data = np.array([1e16, 0, 1, 0, -1e16, 0])
    views = [data[::2], data[::-2], data[:0], data[:1]]
    unaligned = np.ndarray((3,), dtype="d", buffer=bytearray(25), offset=1)
    unaligned[:] = [1e16, 1, -1e16]
    views.append(unaligned)
    for values in views:
        values.setflags(write=False)
        before = values.tobytes()
        result, _ = sum_array(values, backend=backend)
        assert result.to_float() == math.fsum(values)
        assert before == values.tobytes()


def test_zero_copy_buffer_is_forwarded(monkeypatch):
    data = np.array([1e16, 1, -1e16])
    observed = []

    def kernel(values):
        observed.append(values)
        return batch._python_sum(values)

    monkeypatch.setattr(
        batch, "_kernels", SimpleNamespace(API_VERSION=1, neumaier_sum=kernel)
    )
    assert sum_array(data)[0].to_float() == 1.0
    assert observed[0] is data


@pytest.mark.parametrize("module", [None, SimpleNamespace(API_VERSION=999)])
def test_missing_or_incompatible_native_fallback(module, monkeypatch):
    monkeypatch.setattr(batch, "_kernels", module)
    assert not batch.native_available()
    assert sum_array([1e16, 1, -1e16])[0].to_float() == 1
    with pytest.raises(RuntimeError, match="native backend unavailable"):
        sum_array([], backend="native")


def test_backend_errors_are_not_silently_hidden(monkeypatch):
    def broken(values):
        raise RuntimeError("kernel defect")

    monkeypatch.setattr(
        batch, "_kernels", SimpleNamespace(API_VERSION=1, neumaier_sum=broken)
    )
    with pytest.raises(RuntimeError, match="kernel defect"):
        sum_array([1])
    assert sum_array([1], backend="python")[0].to_float() == 1
    with pytest.raises(ValueError, match="backend"):
        sum_array([], backend="unknown")


def test_direct_native_buffer_contract():
    kernels = pytest.importorskip("_balansis_kernels")
    assert kernels.neumaier_sum(array.array("d", [1e16, 1, -1e16])) == (1, 1)
    raw = b"x" + struct.pack("ddd", 1e16, 1, -1e16)
    assert kernels.neumaier_sum(memoryview(raw)[1:].cast("d")) == (1, 1)
    assert kernels.neumaier_sum(array.array("d")) == (0, 0)
    for value in [
        np.ones(3, dtype="f4"),
        np.ones((2, 2)),
        np.ones(6)[::2],
        np.ones(6)[::-1],
        np.ones(3, dtype="d").astype(">f8" if np.little_endian else "<f8"),
        b"12345678",
    ]:
        with pytest.raises(ValueError, match="native float64 buffer"):
            kernels.neumaier_sum(value)
    with pytest.raises(TypeError):
        kernels.neumaier_sum([1.0])
