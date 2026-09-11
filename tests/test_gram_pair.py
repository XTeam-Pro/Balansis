"""Fused Gram arithmetic, dispatch, buffer contracts and complete SVD parity."""

import array
import importlib
import math
import runpy
import struct
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import balansis.array as batch
from balansis import gram_pair

REFERENCE_SVD = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "benchmarks/_svd_before_gram.py")
)["_act_jacobi_svd"]


@pytest.fixture(params=["python", "auto", "native"])
def backend(request):
    if request.param == "native" and not batch.native_gram_available():
        pytest.skip("fused native Gram kernel not installed")
    return request.param


def exact(a, b):
    return float(
        sum((Fraction(float(x)) * Fraction(float(y)) for x, y in zip(a, b)), Fraction())
    )


@pytest.mark.parametrize(
    "a,b",
    [
        ([], []),
        ([0.0, -0.0], [-0.0, 0.0]),
        ([1e16, 1, -1e16], [1, 1, 1]),
        ([math.ulp(0.0)], [-math.ulp(0.0)]),
        ([2.0**-537, 2.0**-537], [2.0**-538, 2.0**-538]),
        ([1e150, 1, -1e150], [1e150, 1, 1e150]),
        ([math.nextafter(1, 2), 1], [math.nextafter(1, 0), -1]),
    ],
)
def test_rational_oracle(a, b, backend):
    expected = exact(a, a), exact(b, b), exact(a, b)
    actual = gram_pair(a, b, backend=backend)
    assert struct.pack("ddd", *actual) == struct.pack("ddd", *expected)


def test_random_extremes_and_permutations(backend):
    rng = np.random.default_rng(941)
    for _ in range(60):
        a = np.ldexp(rng.uniform(-1, 1, 80), rng.integers(-1073, 510, 80))
        b = np.ldexp(rng.uniform(-1, 1, 80), rng.integers(-1073, 510, 80))
        expected = exact(a, a), exact(b, b), exact(a, b)
        actual = gram_pair(a, b, backend=backend)
        assert struct.pack("ddd", *actual) == struct.pack("ddd", *expected)
        order = rng.permutation(len(a))
        assert gram_pair(a[order], b[order], backend=backend) == actual


@pytest.mark.parametrize(
    "a,b", [([1e308], [0]), ([0], [1e308]), ([1e308] * 2, [2, -2])]
)
def test_norm_overflow_rejected_even_when_cross_product_is_finite(a, b, backend):
    with pytest.raises(OverflowError):
        gram_pair(a, b, backend=backend)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_precedes_norm_overflow(bad, backend):
    for a, b in [([1e308], [bad]), ([bad], [1e308])]:
        with pytest.raises(ValueError, match="finite"):
            gram_pair(a, b, backend=backend)


def test_validation_and_conversion(backend):
    for a, b in [([], [1]), ([1, 2], [1])]:
        with pytest.raises(ValueError, match="equal lengths"):
            gram_pair(a, b, backend=backend)
    for a in [3, [[1]]]:
        with pytest.raises(ValueError, match="one-dimensional"):
            gram_pair(a, [1], backend=backend)
    for a in [[1j], ["1"], np.array([1], dtype=object)]:
        with pytest.raises(TypeError, match="real numeric"):
            gram_pair(a, [1], backend=backend)
    for dtype in ["<f8", ">f8", "f4", "i8"]:
        a = np.array([1, 2, 3, 4], dtype=dtype)[::-2]
        assert gram_pair(a, a, backend=backend) == (20, 20, 20)


def test_older_native_kernel_uses_three_exact_calls(monkeypatch):
    calls = []

    def dot(a, b):
        calls.append((a, b))
        return batch._python_dot(a, b)

    monkeypatch.setattr(
        batch, "_kernels", SimpleNamespace(API_VERSION=1, exact_dot=dot)
    )
    assert not batch.native_gram_available()
    assert gram_pair([1, 2], [3, 4]) == (5, 25, 11)
    assert len(calls) == 3
    with pytest.raises(RuntimeError, match="native Gram backend unavailable"):
        gram_pair([1], [1], backend="native")
    with pytest.raises(ValueError, match="backend"):
        gram_pair([], [], backend="invalid")


def test_native_buffer_boundaries_and_release():
    kernels = pytest.importorskip("_balansis_kernels")
    if not hasattr(kernels, "gram_pair"):
        pytest.skip("older native extension")
    a = array.array("d", [1, 2])
    for bad in [
        b"invalid",
        [1, 2],
        array.array("d", [1]),
        np.ones(4)[::2],
        np.ones(2, dtype="f4"),
    ]:
        with pytest.raises((ValueError, TypeError)):
            kernels.gram_pair(a, bad)
        a.append(3)
        a.pop()
    data = np.ndarray((2,), dtype="d", buffer=bytearray(17), offset=1)
    data[:] = [1, 2]
    data.setflags(write=False)
    assert kernels.gram_pair(data, data) == (5, 5, 5)


@pytest.mark.parametrize("shape", [(16, 4), (64, 8), (4, 16), (1, 1), (5, 1)])
def test_svd_bitwise_parity_and_no_input_mutation(shape):
    matrix = np.random.default_rng(193).normal(size=shape)
    matrix.setflags(write=False)
    before = matrix.tobytes()
    expected = REFERENCE_SVD(matrix)
    actual = importlib.import_module("balansis.linalg.svd")._act_jacobi_svd(matrix)
    for old, new in zip(expected, actual):
        assert old.shape == new.shape and old.tobytes() == new.tobytes()
    assert matrix.tobytes() == before
    u, s, vt = actual
    assert np.linalg.norm(u @ np.diag(s) @ vt - matrix) / np.linalg.norm(matrix) < 1e-12


def test_svd_rank_deficient_zero_and_ill_conditioned_parity():
    rng = np.random.default_rng(11)
    q1, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    q2, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    matrices = [
        np.zeros((6, 3)),
        np.ones((8, 3)),
        q1 @ np.diag([1, 1e-5, 1e-10, 1e-15]) @ q2.T,
    ]
    function = importlib.import_module("balansis.linalg.svd")._act_jacobi_svd
    for matrix in matrices:
        for expected, actual in zip(REFERENCE_SVD(matrix), function(matrix)):
            assert expected.tobytes() == actual.tobytes()


@pytest.mark.parametrize("shape", [(32, 4), (4, 32)])
def test_svd_columns_reach_fused_kernel_without_copy(shape, monkeypatch):
    if not batch.native_gram_available():
        pytest.skip("fused native Gram kernel not installed")
    original = batch._kernels.gram_pair
    seen = []

    def record(a, b):
        assert a.flags.c_contiguous and b.flags.c_contiguous
        assert a.base is not None and a.base is b.base
        assert a.base.flags.f_contiguous
        seen.append(id(a.base))
        return original(a, b)

    monkeypatch.setattr(batch._kernels, "gram_pair", record)
    function = importlib.import_module("balansis.linalg.svd")._act_jacobi_svd
    function(np.random.default_rng(39).normal(size=shape))
    assert len(seen) > 1 and len(set(seen)) == 1
