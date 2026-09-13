# Groups, fields and polynomials

`AbsoluteGroup.finite_cyclic_group(n)` constructs addition on the integer
residues `0` through `n-1`. Inversion is negation modulo `n`.
`subgroup(generators)` computes closure, `cosets` enumerates cosets, and
`quotient_group` constructs the quotient of a finite cyclic group.
`additive_group()` and `multiplicative_group()` provide operations on
`AbsoluteValue` elements.

`EternityField.finite_field(p)` constructs a prime field. Addition,
multiplication and inversion use modular arithmetic. Residues are represented
by `FieldElement` objects whose ratios have denominator one.
`EternityField.rational_field()` provides ratio-based operations.

```python
from balansis.algebra.eternity_field import EternityField

field = EternityField.finite_field(5)
for element in field:
    if not element.is_zero():
        assert field.multiply(element, field.multiplicative_inverse(element)) == field.one()
ring = field.polynomial_ring("x")
polynomial = ring.create_polynomial([field.one(), field.one()])
assert polynomial.evaluate(field.one()).ratio.numerical_value() == 2.0
```

Polynomial coefficients are ordered from constant term upward and copied on
construction. Evaluation uses Horner's method. Addition and multiplication
require the same ring. [Source](../balansis/algebra).
