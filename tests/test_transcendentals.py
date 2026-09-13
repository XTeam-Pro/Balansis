import math
import pickle

import pytest

from balansis.core.absolute import AbsoluteValue
from balansis.core.eternity import EternalRatio
from balansis.linalg.gemm import matmul


def test_absolute_transcendentals_basic():
    a = AbsoluteValue.from_float(0.0)
    b = AbsoluteValue.from_float(2.0)
    assert a.sin() == 0.0
    assert a.cos() == 1.0
    assert a.tan() == 0.0
    assert b.log() == math.log(2.0)
    e = AbsoluteValue.from_float(-1.5).exp()
    assert e.is_positive()


def test_absolute_log_non_positive_raises():
    with pytest.raises(ValueError):
        AbsoluteValue.from_float(0.0).log()
    with pytest.raises(ValueError):
        AbsoluteValue.from_float(-3.0).log()


def test_eternity_transcendentals_basic():
    r = EternalRatio.from_values(6.0, 2.0)
    assert r.log() == math.log(3.0)
    r2 = r.exp()
    assert isinstance(r2, EternalRatio)
    assert pytest.approx(r2.numerical_value(), rel=1e-12) == math.exp(3.0)
    assert pytest.approx(r.sin(), rel=1e-12) == math.sin(3.0)
    assert pytest.approx(r.cos(), rel=1e-12) == math.cos(3.0)
    assert pytest.approx(r.tan(), rel=1e-12) == math.tan(3.0)


def test_eternity_log_negative_raises():
    r = EternalRatio.from_values(2.0, -1.0)
    with pytest.raises(ValueError):
        r.log()


def test_pickle_reduce_absolute_eternity_roundtrip():
    a = AbsoluteValue.from_float(-3.5)
    data = pickle.dumps(a)
    a2 = pickle.loads(data)
    assert a2 == a
    r = EternalRatio.from_values(9.0, 3.0)
    data2 = pickle.dumps(r)
    r2 = pickle.loads(data2)
    assert r2 == r


def test_gemm_basic():
    a = [
        [AbsoluteValue.from_float(1.0), AbsoluteValue.from_float(2.0)],
        [AbsoluteValue.from_float(3.0), AbsoluteValue.from_float(4.0)],
    ]
    b = [
        [AbsoluteValue.from_float(5.0), AbsoluteValue.from_float(6.0)],
        [AbsoluteValue.from_float(7.0), AbsoluteValue.from_float(8.0)],
    ]
    c, compensation = matmul(a, b)
    assert c[0][0].to_float() == 1.0 * 5.0 + 2.0 * 7.0
    assert c[0][1].to_float() == 1.0 * 6.0 + 2.0 * 8.0
    assert c[1][0].to_float() == 3.0 * 5.0 + 4.0 * 7.0
    assert c[1][1].to_float() == 3.0 * 6.0 + 4.0 * 8.0
    assert compensation >= 1.0
