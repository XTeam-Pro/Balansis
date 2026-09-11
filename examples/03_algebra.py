from balansis.algebra.eternity_field import EternityField

field = EternityField.finite_field(5)
for element in field:
    if not element.is_zero():
        assert field.multiply(element, field.multiplicative_inverse(element)) == field.one()
ring = field.polynomial_ring("x")
polynomial = ring.create_polynomial([field.one(), field.one()])
assert polynomial.evaluate(field.one()).ratio.numerical_value() == 2.0
