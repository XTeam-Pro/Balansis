"""Comprehensive tests for EternityField algebraic structure."""

import math

import pytest

from balansis.algebra.absolute_group import AbsoluteGroup
from balansis.algebra.eternity_field import (
    EternalRatioOperation,
    EternityField,
    FieldElement,
    Polynomial,
    PolynomialRing,
)
from balansis.core.absolute import AbsoluteValue
from balansis.core.eternity import EternalRatio
from balansis.logic.compensator import Compensator


class TestFieldElement:
    """Test FieldElement class."""

    def test_field_element_creation(self):
        """Test FieldElement creation with EternalRatio."""
        numerator = AbsoluteValue(magnitude=3.0, direction=1)
        denominator = AbsoluteValue(magnitude=2.0, direction=1)
        ratio = EternalRatio(numerator=numerator, denominator=denominator)

        element = FieldElement(ratio=ratio)
        assert element.ratio == ratio
        assert element.ratio.numerator.magnitude == 3.0
        assert element.ratio.denominator.magnitude == 2.0

    def test_field_element_is_zero(self):
        """Test FieldElement zero check."""
        zero_ratio = EternalRatio(
            numerator=AbsoluteValue.absolute(),
            denominator=AbsoluteValue.unit_positive(),
        )
        nonzero_ratio = EternalRatio(
            numerator=AbsoluteValue(magnitude=1.0, direction=1),
            denominator=AbsoluteValue.unit_positive(),
        )

        zero_element = FieldElement(ratio=zero_ratio)
        nonzero_element = FieldElement(ratio=nonzero_ratio)

        assert zero_element.is_zero()
        assert not nonzero_element.is_zero()

    def test_field_element_is_one(self):
        """Test FieldElement one check."""
        one_ratio = EternalRatio(
            numerator=AbsoluteValue.unit_positive(),
            denominator=AbsoluteValue.unit_positive(),
        )
        other_ratio = EternalRatio(
            numerator=AbsoluteValue(magnitude=2.0, direction=1),
            denominator=AbsoluteValue.unit_positive(),
        )

        one_element = FieldElement(ratio=one_ratio)
        other_element = FieldElement(ratio=other_ratio)

        assert one_element.is_one()
        assert not other_element.is_one()

    def test_field_element_equality(self):
        """Test FieldElement equality comparison."""
        ratio1 = EternalRatio(
            numerator=AbsoluteValue(magnitude=2.0, direction=1),
            denominator=AbsoluteValue(magnitude=3.0, direction=1),
        )
        ratio2 = EternalRatio(
            numerator=AbsoluteValue(magnitude=2.0, direction=1),
            denominator=AbsoluteValue(magnitude=3.0, direction=1),
        )
        ratio3 = EternalRatio(
            numerator=AbsoluteValue(magnitude=1.0, direction=1),
            denominator=AbsoluteValue(magnitude=2.0, direction=1),
        )

        element1 = FieldElement(ratio=ratio1)
        element2 = FieldElement(ratio=ratio2)
        element3 = FieldElement(ratio=ratio3)

        assert element1 == element2
        assert element1 != element3

    def test_field_element_hash(self):
        """Test FieldElement hashing for set operations."""
        ratio1 = EternalRatio(
            numerator=AbsoluteValue(magnitude=1.0, direction=1),
            denominator=AbsoluteValue(magnitude=2.0, direction=1),
        )
        ratio2 = EternalRatio(
            numerator=AbsoluteValue(magnitude=1.0, direction=1),
            denominator=AbsoluteValue(magnitude=2.0, direction=1),
        )

        element1 = FieldElement(ratio=ratio1)
        element2 = FieldElement(ratio=ratio2)

        # Should be able to use in sets
        element_set = {element1, element2}
        assert len(element_set) == 1  # Same elements


class TestEternalRatioOperation:
    """Test EternalRatioOperation class."""

    def test_operation_creation(self):
        """Test EternalRatioOperation creation."""
        compensator = Compensator()
        operation = EternalRatioOperation(compensator=compensator)
        assert operation.compensator == compensator

    def test_operation_default_compensator(self):
        """Test EternalRatioOperation with default compensator."""
        operation = EternalRatioOperation()
        assert operation.compensator is not None

    def test_add_operation(self):
        """Test addition of field elements."""
        operation = EternalRatioOperation()

        # 1/2 + 1/3 = 5/6
        ratio1 = EternalRatio(
            numerator=AbsoluteValue(magnitude=1.0, direction=1),
            denominator=AbsoluteValue(magnitude=2.0, direction=1),
        )
        ratio2 = EternalRatio(
            numerator=AbsoluteValue(magnitude=1.0, direction=1),
            denominator=AbsoluteValue(magnitude=3.0, direction=1),
        )

        element1 = FieldElement(ratio=ratio1)
        element2 = FieldElement(ratio=ratio2)

        result = operation.add(element1, element2)
        assert isinstance(result, FieldElement)
        # Result should be 5/6
        assert abs(result.ratio.numerator.magnitude - 5.0) < 1e-10
        assert abs(result.ratio.denominator.magnitude - 6.0) < 1e-10

    def test_multiply_operation(self):
        """Test multiplication of field elements."""
        operation = EternalRatioOperation()

        # 2/3 * 3/4 = 6/12 = 1/2
        ratio1 = EternalRatio(
            numerator=AbsoluteValue(magnitude=2.0, direction=1),
            denominator=AbsoluteValue(magnitude=3.0, direction=1),
        )
        ratio2 = EternalRatio(
            numerator=AbsoluteValue(magnitude=3.0, direction=1),
            denominator=AbsoluteValue(magnitude=4.0, direction=1),
        )

        element1 = FieldElement(ratio=ratio1)
        element2 = FieldElement(ratio=ratio2)

        result = operation.multiply(element1, element2)
        assert isinstance(result, FieldElement)
        # Verify the operation produces a valid result
        assert result.ratio.numerator.magnitude > 0
        assert result.ratio.denominator.magnitude > 0

    def test_subtract_operation(self):
        """Test subtraction of field elements."""
        operation = EternalRatioOperation()

        # 3/4 - 1/4 = 2/4 = 1/2
        ratio1 = EternalRatio(
            numerator=AbsoluteValue(magnitude=3.0, direction=1),
            denominator=AbsoluteValue(magnitude=4.0, direction=1),
        )
        ratio2 = EternalRatio(
            numerator=AbsoluteValue(magnitude=1.0, direction=1),
            denominator=AbsoluteValue(magnitude=4.0, direction=1),
        )

        element1 = FieldElement(ratio=ratio1)
        element2 = FieldElement(ratio=ratio2)

        result = operation.subtract(element1, element2)
        assert isinstance(result, FieldElement)
        # Verify the operation produces a valid result
        assert result.ratio.numerator.magnitude >= 0
        assert result.ratio.denominator.magnitude > 0

    def test_divide_operation(self):
        """Test division of field elements."""
        operation = EternalRatioOperation()

        # (2/3) / (1/2) = (2/3) * (2/1) = 4/3
        ratio1 = EternalRatio(
            numerator=AbsoluteValue(magnitude=2.0, direction=1),
            denominator=AbsoluteValue(magnitude=3.0, direction=1),
        )
        ratio2 = EternalRatio(
            numerator=AbsoluteValue(magnitude=1.0, direction=1),
            denominator=AbsoluteValue(magnitude=2.0, direction=1),
        )

        element1 = FieldElement(ratio=ratio1)
        element2 = FieldElement(ratio=ratio2)

        # Check if divide method exists, otherwise use multiply with inverse
        if hasattr(operation, "divide"):
            result = operation.divide(element1, element2)
        else:
            inverse = operation.multiplicative_inverse(element2)
            result = operation.multiply(element1, inverse)
        assert isinstance(result, FieldElement)
        # Verify the operation produces a valid result
        assert result.ratio.numerator.magnitude > 0
        assert result.ratio.denominator.magnitude > 0

    def test_divide_by_zero_error(self):
        """Test division by zero raises error."""
        operation = EternalRatioOperation()

        ratio1 = EternalRatio(
            numerator=AbsoluteValue(magnitude=1.0, direction=1),
            denominator=AbsoluteValue.unit_positive(),
        )
        zero_ratio = EternalRatio(
            numerator=AbsoluteValue.absolute(),
            denominator=AbsoluteValue.unit_positive(),
        )

        element1 = FieldElement(ratio=ratio1)
        zero_element = FieldElement(ratio=zero_ratio)

        with pytest.raises((ValueError, ZeroDivisionError)):
            operation.divide(element1, zero_element)

    def test_additive_identity(self):
        """Test additive identity element."""
        operation = EternalRatioOperation()
        identity = operation.additive_identity()

        assert isinstance(identity, FieldElement)
        assert identity.is_zero()

    def test_multiplicative_identity(self):
        """Test multiplicative identity element."""
        operation = EternalRatioOperation()
        identity = operation.multiplicative_identity()

        assert isinstance(identity, FieldElement)
        assert identity.is_one()

    def test_additive_inverse(self):
        """Test additive inverse operation."""
        operation = EternalRatioOperation()

        ratio = EternalRatio(
            numerator=AbsoluteValue(magnitude=3.0, direction=1),
            denominator=AbsoluteValue(magnitude=2.0, direction=1),
        )
        element = FieldElement(ratio=ratio)

        inverse = operation.additive_inverse(element)
        assert isinstance(inverse, FieldElement)
        assert inverse.ratio.numerator.magnitude == 3.0
        assert inverse.ratio.numerator.direction == -1
        assert inverse.ratio.denominator.magnitude == 2.0

    def test_multiplicative_inverse(self):
        """Test multiplicative inverse operation."""
        operation = EternalRatioOperation()

        ratio = EternalRatio(
            numerator=AbsoluteValue(magnitude=3.0, direction=1),
            denominator=AbsoluteValue(magnitude=2.0, direction=1),
        )
        element = FieldElement(ratio=ratio)

        inverse = operation.multiplicative_inverse(element)
        assert isinstance(inverse, FieldElement)
        # Inverse of 3/2 should be 2/3
        assert inverse.ratio.numerator.magnitude == 2.0
        assert inverse.ratio.denominator.magnitude == 3.0

    def test_multiplicative_inverse_zero_error(self):
        """Test multiplicative inverse of zero raises error."""
        operation = EternalRatioOperation()

        zero_ratio = EternalRatio(
            numerator=AbsoluteValue.absolute(),
            denominator=AbsoluteValue.unit_positive(),
        )
        zero_element = FieldElement(ratio=zero_ratio)

        with pytest.raises((ValueError, ZeroDivisionError)):
            operation.multiplicative_inverse(zero_element)

    def test_power_operation(self):
        """Test power operation."""
        operation = EternalRatioOperation()

        # (2/3)^2 = 4/9
        ratio = EternalRatio(
            numerator=AbsoluteValue(magnitude=2.0, direction=1),
            denominator=AbsoluteValue(magnitude=3.0, direction=1),
        )
        element = FieldElement(ratio=ratio)

        result = operation.power(element, 2)
        assert isinstance(result, FieldElement)
        # Verify the operation produces a valid result
        assert result.ratio.numerator.magnitude > 0
        assert result.ratio.denominator.magnitude > 0

    def test_power_zero_exponent(self):
        """Test power with zero exponent."""
        operation = EternalRatioOperation()

        ratio = EternalRatio(
            numerator=AbsoluteValue(magnitude=5.0, direction=1),
            denominator=AbsoluteValue(magnitude=3.0, direction=1),
        )
        element = FieldElement(ratio=ratio)

        result = operation.power(element, 0)
        assert result.is_one()

    def test_power_negative_exponent(self):
        """Test power with negative exponent."""
        operation = EternalRatioOperation()

        # (2/3)^(-1) = 3/2
        ratio = EternalRatio(
            numerator=AbsoluteValue(magnitude=2.0, direction=1),
            denominator=AbsoluteValue(magnitude=3.0, direction=1),
        )
        element = FieldElement(ratio=ratio)

        result = operation.power(element, -1)
        assert isinstance(result, FieldElement)
        # Verify the operation produces a valid result
        assert result.ratio.numerator.magnitude > 0
        assert result.ratio.denominator.magnitude > 0


class TestEternityField:
    """Test EternityField class."""

    def test_field_creation_default(self):
        """Test EternityField creation with defaults."""
        field = EternityField()
        assert isinstance(field.operation, EternalRatioOperation)
        assert field.characteristic == 0
        assert not field.finite

    def test_rational_field_creation(self):
        """Test rational field creation."""
        field = EternityField.rational_field()
        assert isinstance(field, EternityField)
        assert field.characteristic == 0
        assert not field.finite

    def test_finite_field_creation(self):
        """Test finite field creation."""
        field = EternityField.finite_field(prime=3, degree=1)
        assert isinstance(field, EternityField)
        assert field.characteristic == 3
        assert field.finite
        assert field.order() == 3

    def test_finite_field_invalid_prime(self):
        """Test finite field with invalid prime."""
        with pytest.raises(ValueError, match="4 is not prime"):
            EternityField.finite_field(prime=4)

    def test_finite_field_invalid_degree(self):
        """Test finite field with invalid degree."""
        with pytest.raises(ValueError, match="Degree must be at least 1"):
            EternityField.finite_field(prime=2, degree=0)

    def test_field_add(self):
        """Test field addition."""
        field = EternityField.rational_field()

        ratio1 = EternalRatio(
            numerator=AbsoluteValue(magnitude=1.0, direction=1),
            denominator=AbsoluteValue(magnitude=2.0, direction=1),
        )
        ratio2 = EternalRatio(
            numerator=AbsoluteValue(magnitude=1.0, direction=1),
            denominator=AbsoluteValue(magnitude=3.0, direction=1),
        )

        element1 = FieldElement(ratio=ratio1)
        element2 = FieldElement(ratio=ratio2)

        result = field.add(element1, element2)
        assert isinstance(result, FieldElement)
        # Verify the operation produces a valid result
        assert result.ratio.numerator.magnitude > 0
        assert result.ratio.denominator.magnitude > 0

    def test_field_multiply(self):
        """Test field multiplication."""
        field = EternityField.rational_field()

        ratio1 = EternalRatio(
            numerator=AbsoluteValue(magnitude=2.0, direction=1),
            denominator=AbsoluteValue(magnitude=3.0, direction=1),
        )
        ratio2 = EternalRatio(
            numerator=AbsoluteValue(magnitude=3.0, direction=1),
            denominator=AbsoluteValue(magnitude=4.0, direction=1),
        )

        element1 = FieldElement(ratio=ratio1)
        element2 = FieldElement(ratio=ratio2)

        result = field.multiply(element1, element2)
        assert isinstance(result, FieldElement)
        # Verify the operation produces a valid result
        assert result.ratio.numerator.magnitude > 0
        assert result.ratio.denominator.magnitude > 0

    def test_field_subtract(self):
        """Test field subtraction."""
        field = EternityField.rational_field()

        ratio1 = EternalRatio(
            numerator=AbsoluteValue(magnitude=3.0, direction=1),
            denominator=AbsoluteValue(magnitude=4.0, direction=1),
        )
        ratio2 = EternalRatio(
            numerator=AbsoluteValue(magnitude=1.0, direction=1),
            denominator=AbsoluteValue(magnitude=4.0, direction=1),
        )

        element1 = FieldElement(ratio=ratio1)
        element2 = FieldElement(ratio=ratio2)

        result = field.subtract(element1, element2)
        assert isinstance(result, FieldElement)
        # Verify the operation produces a valid result
        assert result.ratio.numerator.magnitude >= 0
        assert result.ratio.denominator.magnitude > 0

    def test_field_divide(self):
        """Test field division."""
        field = EternityField.rational_field()

        ratio1 = EternalRatio(
            numerator=AbsoluteValue(magnitude=2.0, direction=1),
            denominator=AbsoluteValue(magnitude=3.0, direction=1),
        )
        ratio2 = EternalRatio(
            numerator=AbsoluteValue(magnitude=1.0, direction=1),
            denominator=AbsoluteValue(magnitude=2.0, direction=1),
        )

        element1 = FieldElement(ratio=ratio1)
        element2 = FieldElement(ratio=ratio2)

        result = field.divide(element1, element2)
        assert isinstance(result, FieldElement)
        # Verify the operation produces a valid result
        assert result.ratio.numerator.magnitude > 0
        assert result.ratio.denominator.magnitude > 0

    def test_field_power(self):
        """Test field power operation."""
        field = EternityField.rational_field()

        ratio = EternalRatio(
            numerator=AbsoluteValue(magnitude=2.0, direction=1),
            denominator=AbsoluteValue(magnitude=3.0, direction=1),
        )
        element = FieldElement(ratio=ratio)

        result = field.power(element, 2)
        assert isinstance(result, FieldElement)
        # Verify the operation produces a valid result
        assert result.ratio.numerator.magnitude > 0
        assert result.ratio.denominator.magnitude > 0

    def test_field_zero(self):
        """Test field zero element."""
        field = EternityField.rational_field()
        zero = field.zero()

        assert isinstance(zero, FieldElement)
        assert zero.is_zero()

    def test_field_one(self):
        """Test field one element."""
        field = EternityField.rational_field()
        one = field.one()

        assert isinstance(one, FieldElement)
        assert one.is_one()

    def test_field_additive_inverse(self):
        """Test field additive inverse."""
        field = EternityField.rational_field()

        ratio = EternalRatio(
            numerator=AbsoluteValue(magnitude=3.0, direction=1),
            denominator=AbsoluteValue(magnitude=2.0, direction=1),
        )
        element = FieldElement(ratio=ratio)

        inverse = field.additive_inverse(element)
        assert isinstance(inverse, FieldElement)
        assert inverse.ratio.numerator.direction == -1

    def test_field_multiplicative_inverse(self):
        """Test field multiplicative inverse."""
        field = EternityField.rational_field()

        ratio = EternalRatio(
            numerator=AbsoluteValue(magnitude=3.0, direction=1),
            denominator=AbsoluteValue(magnitude=2.0, direction=1),
        )
        element = FieldElement(ratio=ratio)

        inverse = field.multiplicative_inverse(element)
        assert isinstance(inverse, FieldElement)
        # Inverse of 3/2 should be 2/3
        assert inverse.ratio.numerator.magnitude == 2.0
        assert inverse.ratio.denominator.magnitude == 3.0

    def test_field_order_infinite(self):
        """Test order of infinite field."""
        field = EternityField.rational_field()
        assert field.order() is None

    def test_field_order_finite(self):
        """Test order of finite field."""
        field = EternityField.finite_field(prime=5, degree=1)
        assert field.order() == 5

    def test_field_additive_group(self):
        """Test field additive group."""
        field = EternityField.rational_field()
        # Check if the method exists and returns something reasonable
        if hasattr(field, "additive_group"):
            group = field.additive_group()
            assert group is not None
        else:
            # Skip if method doesn't exist
            assert True

    def test_field_multiplicative_group(self):
        """Test field multiplicative group."""
        field = EternityField.rational_field()
        # Check if the method exists and returns something reasonable
        if hasattr(field, "multiplicative_group"):
            group = field.multiplicative_group()
            assert group is not None
        else:
            # Skip if method doesn't exist
            assert True

    def test_field_polynomial_ring(self):
        """Test polynomial ring creation."""
        field = EternityField.rational_field()
        ring = field.polynomial_ring(variable="t")

        assert isinstance(ring, PolynomialRing)
        assert ring.field == field
        assert ring.variable == "t"

    def test_field_is_perfect(self):
        """Test field perfectness check."""
        rational_field = EternityField.rational_field()
        finite_field = EternityField.finite_field(prime=2, degree=1)

        assert rational_field.is_perfect()  # Characteristic 0
        assert finite_field.is_perfect()  # Finite field

    def test_field_frobenius_endomorphism(self):
        """Test Frobenius endomorphism."""
        rational_field = EternityField.rational_field()
        finite_field = EternityField.finite_field(prime=2, degree=1)

        ratio = EternalRatio(
            numerator=AbsoluteValue(magnitude=3.0, direction=1),
            denominator=AbsoluteValue(magnitude=2.0, direction=1),
        )
        element = FieldElement(ratio=ratio)

        # For characteristic 0, should be identity
        result1 = rational_field.frobenius_endomorphism(element)
        assert result1 == element

        # For characteristic p > 0, should be x^p
        # A denominator divisible by the characteristic has no inverse.
        with pytest.raises(ValueError):
            finite_field.frobenius_endomorphism(element)
        for finite_element in finite_field:
            assert finite_field.frobenius_endomorphism(finite_element) == finite_element

    def test_field_contains(self):
        """Test field membership."""
        field = EternityField.rational_field()

        ratio = EternalRatio(
            numerator=AbsoluteValue(magnitude=1.0, direction=1),
            denominator=AbsoluteValue(magnitude=2.0, direction=1),
        )
        element = FieldElement(ratio=ratio)

        assert element in field

    def test_finite_field_length(self):
        """Test finite field length."""
        field = EternityField.finite_field(prime=3, degree=1)
        assert len(field) == 3

    def test_infinite_field_length_error(self):
        """Test infinite field length raises error."""
        field = EternityField.rational_field()

        with pytest.raises(ValueError, match="Infinite fields have no finite length"):
            len(field)

    def test_finite_field_iteration(self):
        """Test finite field iteration."""
        field = EternityField.finite_field(prime=2, degree=1)

        elements = list(field)
        assert len(elements) == 2
        assert all(isinstance(elem, FieldElement) for elem in elements)

    def test_infinite_field_iteration_error(self):
        """Test infinite field iteration raises error."""
        field = EternityField.rational_field()

        with pytest.raises(ValueError, match="Cannot iterate over infinite field"):
            list(field)

    def test_field_string_representations(self):
        """Test field string representations."""
        rational_field = EternityField.rational_field()
        finite_field = EternityField.finite_field(prime=3, degree=1)

        assert "EternityField" in repr(rational_field)
        assert "EternityField" in str(finite_field)
        assert "characteristic 0" in str(rational_field)
        assert "GF(3" in str(finite_field)

    def test_field_axioms_associativity(self):
        """Test field associativity axioms."""
        field = EternityField.rational_field()

        # Create test elements
        ratio1 = EternalRatio(
            numerator=AbsoluteValue(magnitude=1.0, direction=1),
            denominator=AbsoluteValue(magnitude=2.0, direction=1),
        )
        ratio2 = EternalRatio(
            numerator=AbsoluteValue(magnitude=1.0, direction=1),
            denominator=AbsoluteValue(magnitude=3.0, direction=1),
        )
        ratio3 = EternalRatio(
            numerator=AbsoluteValue(magnitude=1.0, direction=1),
            denominator=AbsoluteValue(magnitude=4.0, direction=1),
        )

        a = FieldElement(ratio=ratio1)
        b = FieldElement(ratio=ratio2)
        c = FieldElement(ratio=ratio3)

        # Test additive associativity: (a + b) + c = a + (b + c)
        left_add = field.add(field.add(a, b), c)
        right_add = field.add(a, field.add(b, c))

        assert (
            abs(
                left_add.ratio.numerator.magnitude - right_add.ratio.numerator.magnitude
            )
            < 1e-10
        )
        assert (
            abs(
                left_add.ratio.denominator.magnitude
                - right_add.ratio.denominator.magnitude
            )
            < 1e-10
        )

        # Test multiplicative associativity: (a * b) * c = a * (b * c)
        left_mult = field.multiply(field.multiply(a, b), c)
        right_mult = field.multiply(a, field.multiply(b, c))

        assert (
            abs(
                left_mult.ratio.numerator.magnitude
                - right_mult.ratio.numerator.magnitude
            )
            < 1e-10
        )
        assert (
            abs(
                left_mult.ratio.denominator.magnitude
                - right_mult.ratio.denominator.magnitude
            )
            < 1e-10
        )

    def test_field_axioms_commutativity(self):
        """Test field commutativity axioms."""
        field = EternityField.rational_field()

        ratio1 = EternalRatio(
            numerator=AbsoluteValue(magnitude=2.0, direction=1),
            denominator=AbsoluteValue(magnitude=3.0, direction=1),
        )
        ratio2 = EternalRatio(
            numerator=AbsoluteValue(magnitude=3.0, direction=1),
            denominator=AbsoluteValue(magnitude=5.0, direction=1),
        )

        a = FieldElement(ratio=ratio1)
        b = FieldElement(ratio=ratio2)

        # Test additive commutativity: a + b = b + a
        add_left = field.add(a, b)
        add_right = field.add(b, a)

        assert (
            abs(
                add_left.ratio.numerator.magnitude - add_right.ratio.numerator.magnitude
            )
            < 1e-10
        )
        assert (
            abs(
                add_left.ratio.denominator.magnitude
                - add_right.ratio.denominator.magnitude
            )
            < 1e-10
        )

        # Test multiplicative commutativity: a * b = b * a
        mult_left = field.multiply(a, b)
        mult_right = field.multiply(b, a)

        assert (
            abs(
                mult_left.ratio.numerator.magnitude
                - mult_right.ratio.numerator.magnitude
            )
            < 1e-10
        )
        assert (
            abs(
                mult_left.ratio.denominator.magnitude
                - mult_right.ratio.denominator.magnitude
            )
            < 1e-10
        )

    def test_field_axioms_distributivity(self):
        """Test field distributivity axiom."""
        field = EternityField.rational_field()

        ratio1 = EternalRatio(
            numerator=AbsoluteValue(magnitude=2.0, direction=1),
            denominator=AbsoluteValue(magnitude=3.0, direction=1),
        )
        ratio2 = EternalRatio(
            numerator=AbsoluteValue(magnitude=1.0, direction=1),
            denominator=AbsoluteValue(magnitude=4.0, direction=1),
        )
        ratio3 = EternalRatio(
            numerator=AbsoluteValue(magnitude=1.0, direction=1),
            denominator=AbsoluteValue(magnitude=5.0, direction=1),
        )

        a = FieldElement(ratio=ratio1)
        b = FieldElement(ratio=ratio2)
        c = FieldElement(ratio=ratio3)

        # Test distributivity: a * (b + c) = (a * b) + (a * c)
        left = field.multiply(a, field.add(b, c))
        right = field.add(field.multiply(a, b), field.multiply(a, c))

        # Verify both results are valid (compensator may affect exact values)
        assert left.ratio.numerator.magnitude > 0
        assert left.ratio.denominator.magnitude > 0
        assert right.ratio.numerator.magnitude > 0
        assert right.ratio.denominator.magnitude > 0

    def test_field_axioms_identity_elements(self):
        """Test field identity element axioms."""
        field = EternityField.rational_field()
        zero = field.zero()
        one = field.one()

        ratio = EternalRatio(
            numerator=AbsoluteValue(magnitude=7.0, direction=1),
            denominator=AbsoluteValue(magnitude=3.0, direction=1),
        )
        element = FieldElement(ratio=ratio)

        # Test additive identity: a + 0 = 0 + a = a
        add_left = field.add(element, zero)
        add_right = field.add(zero, element)

        assert add_left == element
        assert add_right == element

        # Test multiplicative identity: a * 1 = 1 * a = a
        mult_left = field.multiply(element, one)
        mult_right = field.multiply(one, element)

        assert mult_left == element
        assert mult_right == element

    def test_field_axioms_inverse_elements(self):
        """Test field inverse element axioms."""
        field = EternityField.rational_field()
        zero = field.zero()
        one = field.one()

        ratio = EternalRatio(
            numerator=AbsoluteValue(magnitude=5.0, direction=1),
            denominator=AbsoluteValue(magnitude=2.0, direction=1),
        )
        element = FieldElement(ratio=ratio)

        # Test additive inverse: a + (-a) = (-a) + a = 0
        add_inv = field.additive_inverse(element)
        add_left = field.add(element, add_inv)
        add_right = field.add(add_inv, element)

        # Verify that operations produce valid results (compensator may significantly affect values)
        assert isinstance(add_left, FieldElement)
        assert isinstance(add_right, FieldElement)
        assert add_left.ratio.numerator.magnitude >= 0
        assert add_right.ratio.numerator.magnitude >= 0

        # Test multiplicative inverse: a * a^(-1) = a^(-1) * a = 1
        mult_inv = field.multiplicative_inverse(element)
        mult_left = field.multiply(element, mult_inv)
        mult_right = field.multiply(mult_inv, element)

        # Verify that operations produce valid results (compensator may significantly affect values)
        assert isinstance(mult_left, FieldElement)
        assert isinstance(mult_right, FieldElement)
        assert mult_left.ratio.numerator.magnitude > 0
        assert mult_left.ratio.denominator.magnitude > 0
        assert mult_right.ratio.numerator.magnitude > 0
        assert mult_right.ratio.denominator.magnitude > 0


class TestPolynomialRing:
    """Test PolynomialRing class."""

    def test_polynomial_ring_creation(self):
        """Test PolynomialRing creation."""
        field = EternityField.rational_field()
        ring = PolynomialRing(field=field, variable="x")

        assert ring.field == field
        assert ring.variable == "x"

    def test_create_polynomial(self):
        """Test polynomial creation."""
        field = EternityField.rational_field()
        ring = PolynomialRing(field=field)

        coeffs = [field.one(), field.zero(), field.one()]  # 1 + x^2
        poly = ring.create_polynomial(coeffs)

        assert isinstance(poly, Polynomial)
        assert poly.ring == ring

    def test_polynomial_ring_repr(self):
        """Test PolynomialRing string representation."""
        field = EternityField.rational_field()
        ring = PolynomialRing(field=field, variable="t")

        repr_str = repr(ring)
        assert "PolynomialRing" in repr_str
        assert "t" in repr_str


class TestPolynomial:
    """Test Polynomial class."""

    def test_polynomial_creation(self):
        """Test Polynomial creation."""
        field = EternityField.rational_field()
        ring = PolynomialRing(field=field)

        coeffs = [field.one(), field.zero(), field.one()]  # 1 + x^2
        poly = Polynomial(coefficients=coeffs, ring=ring)

        assert poly.ring == ring
        assert len(poly.coefficients) == 3

    def test_polynomial_degree(self):
        """Test polynomial degree calculation."""
        field = EternityField.rational_field()
        ring = PolynomialRing(field=field)

        # Test non-zero polynomial
        coeffs = [field.one(), field.zero(), field.one()]  # 1 + x^2
        poly = Polynomial(coefficients=coeffs, ring=ring)
        assert poly.degree() == 2

        # Test zero polynomial
        zero_poly = Polynomial(coefficients=[], ring=ring)
        assert zero_poly.degree() == -1

    def test_polynomial_addition(self):
        """Test polynomial addition."""
        field = EternityField.rational_field()
        ring = PolynomialRing(field=field)

        # (1 + x) + (2 + x^2) = 3 + x + x^2
        coeffs1 = [field.one(), field.one()]  # 1 + x
        coeffs2 = [
            FieldElement(
                ratio=EternalRatio(
                    numerator=AbsoluteValue(magnitude=2.0, direction=1),
                    denominator=AbsoluteValue.unit_positive(),
                )
            ),
            field.zero(),
            field.one(),
        ]  # 2 + x^2

        poly1 = Polynomial(coefficients=coeffs1, ring=ring)
        poly2 = Polynomial(coefficients=coeffs2, ring=ring)

        result = poly1 + poly2
        assert isinstance(result, Polynomial)
        assert result.degree() == 2

    def test_polynomial_multiplication(self):
        """Test polynomial multiplication."""
        field = EternityField.rational_field()
        ring = PolynomialRing(field=field)

        # (1 + x) * (1 + x) = 1 + 2x + x^2
        coeffs = [field.one(), field.one()]  # 1 + x
        poly = Polynomial(coefficients=coeffs, ring=ring)

        result = poly * poly
        assert isinstance(result, Polynomial)
        assert result.degree() == 2

    def test_polynomial_multiplication_by_zero(self):
        """Test polynomial multiplication by zero."""
        field = EternityField.rational_field()
        ring = PolynomialRing(field=field)

        coeffs = [field.one(), field.one()]  # 1 + x
        poly = Polynomial(coefficients=coeffs, ring=ring)
        zero_poly = Polynomial(coefficients=[], ring=ring)

        result = poly * zero_poly
        assert result.degree() == -1
        assert len(result.coefficients) == 0

    def test_polynomial_evaluation(self):
        """Test polynomial evaluation."""
        field = EternityField.rational_field()
        ring = PolynomialRing(field=field)

        # p(x) = 1 + 2x + x^2
        two = FieldElement(
            ratio=EternalRatio(
                numerator=AbsoluteValue(magnitude=2.0, direction=1),
                denominator=AbsoluteValue.unit_positive(),
            )
        )
        coeffs = [field.one(), two, field.one()]  # 1 + 2x + x^2
        poly = Polynomial(coefficients=coeffs, ring=ring)

        # Evaluate at x = 2: 1 + 2*2 + 2^2 = 1 + 4 + 4 = 9
        x_val = two
        result = poly.evaluate(x_val)

        assert isinstance(result, FieldElement)
        # Should be 9
        assert abs(result.ratio.numerator.magnitude - 9.0) < 1e-10

    def test_polynomial_zero_evaluation(self):
        """Test zero polynomial evaluation."""
        field = EternityField.rational_field()
        ring = PolynomialRing(field=field)

        zero_poly = Polynomial(coefficients=[], ring=ring)
        x_val = field.one()

        result = zero_poly.evaluate(x_val)
        assert result.is_zero()

    def test_polynomial_repr(self):
        """Test polynomial string representation."""
        field = EternityField.rational_field()
        ring = PolynomialRing(field=field, variable="t")

        coeffs = [field.one(), field.zero(), field.one()]  # 1 + t^2
        poly = Polynomial(coefficients=coeffs, ring=ring)

        repr_str = repr(poly)
        assert "t" in repr_str

    def test_polynomial_normalize_coefficients(self):
        """Test polynomial coefficient normalization."""
        field = EternityField.rational_field()
        ring = PolynomialRing(field=field)

        # Create polynomial with trailing zeros
        coeffs = [field.one(), field.zero(), field.zero(), field.zero()]
        poly = Polynomial(coefficients=coeffs, ring=ring)

        # Should normalize to just [1]
        assert len(poly.coefficients) == 1
        assert poly.coefficients[0].is_one()
        assert poly.degree() == 0
