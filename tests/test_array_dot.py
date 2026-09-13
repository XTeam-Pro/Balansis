"""Independent rational oracles for the exact binary64 dot product."""

import array
import importlib
import math
import struct
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

import balansis.array as batch
from balansis import dot_array
from balansis.core._eft import dot2
from balansis.numpy_integration import compensated_dot_product


@pytest.fixture(params=["python", "native", "auto"])
def backend(request):
    if request.param == "native" and not batch.native_dot_available():
        pytest.skip("native exact-dot extension not installed")
    return request.param


def oracle(a, b):
    return float(
        sum((Fraction(float(x)) * Fraction(float(y)) for x, y in zip(a, b)), Fraction())
    )


def assert_bits(actual, expected):
    assert struct.pack("d", actual) == struct.pack("d", expected)


@pytest.mark.parametrize(
    "a,b",
    [
        ([], []),
        ([-0.0], [1.0]),
        ([1e16, 1, -1e16], [1, 1, 1]),
        ([1e308, 1e308], [2, -2]),
        ([np.finfo(float).max] * 3, [np.finfo(float).max, -np.finfo(float).max, 1]),
        ([math.ulp(0.0)] * 2, [0.5, 0.5]),
        ([math.ulp(0.0)], [0.5]),
        ([-math.ulp(0.0)], [0.5]),
        ([math.ulp(0.0)] * 3, [0.5, 0.5, 0.5]),
        ([math.ulp(0.0)] * 2, [0.5, math.ulp(0.0)]),
        ([1, 2.0**-53], [1, 1]),
        ([1, 2.0**-53, math.ulp(0.0)], [1, 1, 1]),
        ([math.nextafter(1, 2), 2.0**-53], [1, 1]),
        ([2.0**-1022, -math.ulp(0.0)], [1, 0.5]),
        ([1e308, 1e-308], [1e-308, -1e308]),
        ([np.finfo(float).max, 2.0**969], [1, 1]),
        ([math.nextafter(1, 2), -1], [math.nextafter(1, 0), 1]),
    ],
)
def test_exact_boundaries(a, b, backend):
    assert_bits(dot_array(a, b, backend=backend), oracle(a, b))


@pytest.mark.parametrize("sign", [1, -1])
def test_final_overflow(sign, backend):
    for a, b in [
        ([sign * np.finfo(float).max], [2]),
        ([sign * np.finfo(float).max, sign * 2.0**970], [1, 1]),
    ]:
        with pytest.raises(OverflowError):
            oracle(a, b)
        with pytest.raises(OverflowError):
            dot_array(a, b, backend=backend)


def test_seeded_all_exponents_and_permutations(backend):
    rng = np.random.default_rng(2831)
    for _ in range(120):
        a = rng.integers(0, 2**64, size=64, dtype=np.uint64).view(np.float64)
        b = rng.integers(0, 2**64, size=64, dtype=np.uint64).view(np.float64)
        a[~np.isfinite(a)] = 0
        b[~np.isfinite(b)] = 0
        try:
            expected = oracle(a, b)
        except OverflowError:
            with pytest.raises(OverflowError):
                dot_array(a, b, backend=backend)
            continue
        assert_bits(dot_array(a, b, backend=backend), expected)
        order = rng.permutation(len(a))
        assert_bits(dot_array(a[order], b[order], backend=backend), expected)


def test_exact_cancellation_across_exponent_range(backend):
    rng = np.random.default_rng(399)
    for _ in range(40):
        a = np.ldexp(rng.uniform(0.5, 1, 80), rng.integers(-1073, 1024, 80))
        b = np.ldexp(rng.uniform(0.5, 1, 80), rng.integers(-1073, 1024, 80))
        left = np.concatenate([a, -a, [math.ulp(0.0), math.ulp(0.0)]])
        right = np.concatenate([b, b, [0.5, 0.5]])
        order = rng.permutation(len(left))
        assert_bits(
            dot_array(left[order], right[order], backend=backend), math.ulp(0.0)
        )


def test_rounding_ties_and_sticky_bits_at_every_binary64_exponent(backend):
    tiny = math.ulp(0.0)
    for exponent in range(-1074, 1024):
        base = math.ldexp(1.0, exponent)
        unit = math.ulp(base)
        for a, b in [
            ([base, unit], [1.0, 0.5]),
            ([base, unit, tiny], [1.0, 0.5, tiny]),
            ([-math.nextafter(base, math.inf), -unit], [1.0, 0.5]),
        ]:
            assert_bits(dot_array(a, b, backend=backend), oracle(a, b))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("side", [0, 1])
def test_nonfinite_rejected_including_zero_product(bad, side, backend):
    args = [[0.0], [0.0]]
    args[side] = [bad]
    with pytest.raises(ValueError, match="finite"):
        dot_array(*args, backend=backend)


@pytest.mark.parametrize("a,b", [([], [1]), ([1], [1, 2]), ([1, 2], [1])])
def test_lengths_rejected(a, b, backend):
    with pytest.raises(ValueError, match="equal lengths"):
        dot_array(a, b, backend=backend)


def test_input_conversion_and_views(backend):
    for dtype in ["<f8", ">f8", "f4", "i8"]:
        a = np.array([3, -1, 2, 4], dtype=dtype)[::-1]
        b = np.array([7, 8, 9, 10], dtype=dtype)[::-1]
        a.setflags(write=False)
        b.setflags(write=False)
        assert_bits(dot_array(a, b, backend=backend), oracle(a, b))
    a = np.ndarray((3,), dtype="d", buffer=bytearray(25), offset=1)
    a[:] = [1e16, 1, -1e16]
    assert dot_array(a, np.ones(3), backend=backend) == 1
    for values in [[1j], ["1"], np.array([1], dtype=object)]:
        with pytest.raises(TypeError):
            dot_array(values, [1], backend=backend)
    for values in [1, [[1]]]:
        with pytest.raises(ValueError, match="one-dimensional"):
            dot_array(values, [1], backend=backend)


def test_zero_copy_and_older_extension(monkeypatch):
    a, b = np.ones(3), np.ones(3)
    seen = []

    def kernel(x, y):
        seen.append((x, y))
        return 3.0

    monkeypatch.setattr(
        batch, "_kernels", SimpleNamespace(API_VERSION=1, exact_dot=kernel)
    )
    assert dot_array(a, b) == 3
    assert seen[0][0] is a and seen[0][1] is b
    monkeypatch.setattr(batch, "_kernels", SimpleNamespace(API_VERSION=1))
    assert batch.native_available() and not batch.native_dot_available()
    assert dot_array(a, b) == 3
    with pytest.raises(RuntimeError, match="native dot backend unavailable"):
        dot_array(a, b, backend="native")
    with pytest.raises(ValueError, match="backend"):
        dot_array(a, b, backend="unknown")


def test_dot2_and_numpy_adapter_use_exact_backend():
    tiny = math.ulp(0.0)
    for function in [dot2, compensated_dot_product]:
        assert function([tiny, tiny], [0.5, 0.5]) == tiny
        assert function([1e308, 1e308], [2, -2]) == 0
        assert function([[1, 2]], [[3, 4]]) == 11
        with pytest.raises(ValueError, match="equal lengths"):
            function([], [1])
        assert math.isinf(function([math.inf], [1]))


def test_native_buffer_release_on_error():
    kernels = pytest.importorskip("_balansis_kernels")
    if not hasattr(kernels, "exact_dot"):
        pytest.skip("older sum-only native extension")
    a = array.array("d", [1, 2])
    for b in [b"invalid", [1, 2], array.array("d", [1])]:
        with pytest.raises((ValueError, TypeError)):
            kernels.exact_dot(a, b)
        a.append(3)  # A leaked Py_buffer export would prohibit resizing.
        a.pop()
    for bad in [np.ones(6)[::2], np.ones((2, 2)), np.ones(2, dtype="f4")]:
        with pytest.raises(ValueError):
            kernels.exact_dot(bad, bad)
    assert kernels.exact_dot(memoryview(a), memoryview(a)) == 5


def test_svd_dispatches_to_native_and_preserves_reconstruction(monkeypatch):
    if not batch.native_dot_available():
        pytest.skip("native exact-dot extension not installed")
    kernels = batch._kernels
    seen = []

    def recording(a, b):
        seen.append(len(a))
        return kernels.exact_dot(a, b)

    monkeypatch.setattr(
        batch, "_kernels", SimpleNamespace(API_VERSION=1, exact_dot=recording)
    )
    module = importlib.import_module("balansis.linalg.svd")
    matrix = np.random.default_rng(32).normal(size=(12, 4))
    u, s, vt = module._act_jacobi_svd(matrix)
    assert seen and set(seen) == {12}
    assert np.linalg.norm(u @ np.diag(s) @ vt - matrix) < 1e-12
