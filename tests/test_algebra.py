import pytest

from balansis import EternalRatio
from balansis.algebra.eternity_field import EternityField, FieldElement


@pytest.mark.parametrize("value", [-3.0, -1.0, 0.0, 1.0, 3.0])
def test_polynomial_evaluation_and_arithmetic(value):
    field = EternityField.rational_field()
    ring = field.polynomial_ring()
    one = field.one()
    polynomial = ring.create_polynomial([one, one])
    argument = FieldElement(ratio=EternalRatio.from_float(value))
    assert polynomial.evaluate(argument).ratio.numerical_value() == value + 1
    assert (polynomial + polynomial).evaluate(argument).ratio.numerical_value() == 2 * (
        value + 1
    )
    assert (polynomial * polynomial).evaluate(argument).ratio.numerical_value() == (
        value + 1
    ) ** 2
