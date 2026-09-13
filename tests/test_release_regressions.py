import importlib
import itertools
import math
import subprocess
import sys
from decimal import Decimal

import numpy as np
import pytest

from balansis import AbsoluteValue, B, EternalRatio, Operations
from balansis.algebra.absolute_group import AbsoluteGroup, GroupElement
from balansis.algebra.eternity_field import EternityField, FieldElement
from balansis.logic.compensator import Compensator
from balansis.sets import EternalSet, global_compensate
from balansis.sets.resolver import stream_compensate
from balansis.vectorized import batch_add


@pytest.mark.parametrize("value", [-10.0, -1.0, 0.0, 1.0, 10.0])
def test_exponential_matches_math(value):
    assert B(value).exp().to_float() == math.exp(value)


@pytest.mark.parametrize("value", [0.0, -0.0, 1.0, -1.0, 2.0, 1e-300])
def test_numeric_hash_contract(value):
    assert B(value) == value
    assert hash(B(value)) == hash(value)
    assert {B(value): "value"}[value] == "value"


@pytest.mark.parametrize("value", [1e-320, -1e-320, 1e308, -1e308])
def test_division_does_not_overflow_reciprocal(value):
    assert (B(value) / value).to_float() == 1.0


@pytest.mark.parametrize(
    "base, exponent", [(-2.0, -3), (-2.0, -2), (-2.0, 3), (4.0, 0.5), (4.0, -0.5)]
)
def test_ratio_power_sign(base, exponent):
    assert (
        EternalRatio.from_float(base) ** exponent
    ).numerical_value() == base**exponent


def test_nonreal_power_rejected():
    with pytest.raises(ValueError):
        EternalRatio.from_float(-2.0) ** 0.5


def test_import_does_not_change_decimal_precision():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from decimal import getcontext; getcontext().prec=17; import balansis; import balansis.finance; assert getcontext().prec==17",
        ],
        check=True,
    )


def test_stream_emits_each_pair_once():
    a = EternalSet([B(1), B(2)], is_infinite=True)
    b = EternalSet([B(-1), B(3)], is_infinite=True)
    assert [x.to_float() for x in global_compensate(a, b)] == [0.0, 5.0]
    assert list(stream_compensate([B(1)], [], limit=0)) == []
    with pytest.raises(ValueError):
        batch_add([B(1)], [])


def test_stability_for_large_magnitudes():
    c = Compensator()
    assert math.isfinite(c.analyze_stability([B(1e308), B(5e307)]))
    assert c.compensate_power(B(1e-100), 10).is_absolute()


@pytest.mark.parametrize("order", [1, 2, 6, 9])
def test_cyclic_group_exhaustive_laws(order):
    group = AbsoluteGroup.finite_cyclic_group(order)
    carrier = set(group.elements)
    zero = group.identity_element()
    for a, b, c in itertools.product(carrier, repeat=3):
        assert group.operate(a, b) in carrier
        assert group.operate(group.operate(a, b), c) == group.operate(
            a, group.operate(b, c)
        )
    for a in carrier:
        assert group.operate(a, group.inverse_element(a)) == zero
    assert group.elements == carrier
    assert (
        importlib.import_module("balansis.algebra.group").GroupElement is GroupElement
    )


@pytest.mark.parametrize("prime", [2, 3, 5, 7])
def test_prime_field_exhaustive_laws(prime):
    field = EternityField.finite_field(prime)
    carrier = list(field)
    for a, b, c in itertools.product(carrier, repeat=3):
        assert field.add(a, b) in field
        assert field.multiply(a, b) in field
        assert field.multiply(a, field.add(b, c)) == field.add(
            field.multiply(a, b), field.multiply(a, c)
        )
    for a in carrier:
        assert field.add(a, field.additive_inverse(a)) == field.zero()
        if not a.is_zero():
            assert field.multiply(a, field.multiplicative_inverse(a)) == field.one()
    assert len(field.additive_group()) == prime
    assert len(field.multiplicative_group()) == prime - 1


def test_polynomial_owns_its_coefficients():
    field = EternityField.rational_field()
    coefficients = [field.one(), field.zero()]
    polynomial = field.polynomial_ring().create_polynomial(coefficients)
    assert len(coefficients) == 2
    coefficients[0] = field.zero()
    assert polynomial.evaluate(field.one()) == field.one()


@pytest.mark.parametrize("scale", [1e-200, 1e-100, 1e100, 1e200])
def test_scaled_jacobi_svd(scale):
    function = importlib.import_module("balansis.linalg.svd")._act_jacobi_svd
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    u, s, vt = function(a * scale)
    assert np.allclose(u @ np.diag(s / scale) @ vt, a, atol=1e-12)
    assert np.allclose(u.T @ u, np.eye(2), atol=1e-12)


def test_benchmark_package_import():
    import benchmarks

    assert callable(benchmarks.LinalgBenchmark)
