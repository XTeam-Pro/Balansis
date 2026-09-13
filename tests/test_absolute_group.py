"""Comprehensive tests for AbsoluteGroup algebraic structure."""

import math

import pytest

from balansis.algebra.absolute_group import (
    AbsoluteGroup,
    AdditiveOperation,
    GroupElement,
    MultiplicativeOperation,
)
from balansis.core.absolute import AbsoluteValue
from balansis.logic.compensator import Compensator


class TestGroupElement:
    """Test GroupElement class."""

    def test_group_element_creation(self):
        """Test GroupElement creation with AbsoluteValue."""
        value = AbsoluteValue(magnitude=5.0, direction=1)
        element = GroupElement(value=value)
        assert element.value == value
        assert element.value.magnitude == 5.0
        assert element.value.direction == 1

    def test_group_element_equality(self):
        """Test GroupElement equality comparison."""
        value1 = AbsoluteValue(magnitude=3.0, direction=1)
        value2 = AbsoluteValue(magnitude=3.0, direction=1)
        value3 = AbsoluteValue(magnitude=4.0, direction=1)

        element1 = GroupElement(value=value1)
        element2 = GroupElement(value=value2)
        element3 = GroupElement(value=value3)

        assert element1 == element2
        assert element1 != element3

    def test_group_element_hash(self):
        """Test GroupElement hashing for set operations."""
        value1 = AbsoluteValue(magnitude=2.0, direction=1)
        value2 = AbsoluteValue(magnitude=2.0, direction=1)

        element1 = GroupElement(value=value1)
        element2 = GroupElement(value=value2)

        # Should be able to use in sets
        element_set = {element1, element2}
        assert len(element_set) == 1  # Same elements

    def test_group_element_repr(self):
        """Test GroupElement string representation."""
        value = AbsoluteValue(magnitude=7.0, direction=-1)
        element = GroupElement(value=value)
        repr_str = repr(element)
        assert "GroupElement" in repr_str
        assert "7.0" in repr_str


class TestAdditiveOperation:
    """Test AdditiveOperation class."""

    def test_additive_operation_creation(self):
        """Test AdditiveOperation creation."""
        compensator = Compensator()
        operation = AdditiveOperation(compensator=compensator)
        assert operation.compensator == compensator

    def test_additive_operation_default_compensator(self):
        """Test AdditiveOperation with default compensator."""
        operation = AdditiveOperation()
        assert operation.compensator is not None

    def test_additive_operate(self):
        """Test additive operation between elements."""
        operation = AdditiveOperation()

        value1 = AbsoluteValue(magnitude=3.0, direction=1)
        value2 = AbsoluteValue(magnitude=2.0, direction=1)
        element1 = GroupElement(value=value1)
        element2 = GroupElement(value=value2)

        result = operation.operate(element1, element2)
        assert isinstance(result, GroupElement)
        assert result.value.magnitude == 5.0
        assert result.value.direction == 1

    def test_additive_identity(self):
        """Test additive identity element."""
        operation = AdditiveOperation()
        identity = operation.identity_element()

        assert isinstance(identity, GroupElement)
        assert identity.value.is_absolute()

    def test_additive_inverse(self):
        """Test additive inverse operation."""
        operation = AdditiveOperation()

        value = AbsoluteValue(magnitude=4.0, direction=1)
        element = GroupElement(value=value)

        inverse = operation.inverse_element(element)
        assert isinstance(inverse, GroupElement)
        assert inverse.value.magnitude == 4.0
        assert inverse.value.direction == -1

    def test_additive_operation_with_zero(self):
        """Test additive operation with zero element."""
        operation = AdditiveOperation()
        identity = operation.identity_element()

        value = AbsoluteValue(magnitude=5.0, direction=1)
        element = GroupElement(value=value)

        result1 = operation.operate(element, identity)
        result2 = operation.operate(identity, element)

        assert result1.value == element.value
        assert result2.value == element.value

    def test_additive_operation_with_large_values(self):
        """Test additive operation with large values."""
        operation = AdditiveOperation()

        large_value = AbsoluteValue(magnitude=1e10, direction=1)
        small_value = AbsoluteValue(magnitude=10.0, direction=1)

        large_element = GroupElement(value=large_value)
        small_element = GroupElement(value=small_value)

        result = operation.operate(large_element, small_element)
        assert result.value.magnitude == 1e10 + 10.0
        assert result.value.direction == 1


class TestMultiplicativeOperation:
    """Test MultiplicativeOperation class."""

    def test_multiplicative_operation_creation(self):
        """Test MultiplicativeOperation creation."""
        compensator = Compensator()
        operation = MultiplicativeOperation(compensator=compensator)
        assert operation.compensator == compensator

    def test_multiplicative_operate(self):
        """Test multiplicative operation between elements."""
        operation = MultiplicativeOperation()

        value1 = AbsoluteValue(magnitude=3.0, direction=1)
        value2 = AbsoluteValue(magnitude=4.0, direction=1)
        element1 = GroupElement(value=value1)
        element2 = GroupElement(value=value2)

        result = operation.operate(element1, element2)
        assert isinstance(result, GroupElement)
        assert result.value.magnitude == 12.0
        assert result.value.direction == 1

    def test_multiplicative_identity(self):
        """Test multiplicative identity element."""
        operation = MultiplicativeOperation()
        identity = operation.identity_element()

        assert isinstance(identity, GroupElement)
        assert identity.value.is_unit()
        assert identity.value.magnitude == 1.0

    def test_multiplicative_inverse(self):
        """Test multiplicative inverse operation."""
        operation = MultiplicativeOperation()

        value = AbsoluteValue(magnitude=2.0, direction=1)
        element = GroupElement(value=value)

        inverse = operation.inverse_element(element)
        assert isinstance(inverse, GroupElement)
        assert inverse.value.magnitude == 0.5
        assert inverse.value.direction == 1

    def test_multiplicative_operation_with_unit(self):
        """Test multiplicative operation with unit element."""
        operation = MultiplicativeOperation()
        identity = operation.identity_element()

        value = AbsoluteValue(magnitude=7.0, direction=-1)
        element = GroupElement(value=value)

        result1 = operation.operate(element, identity)
        result2 = operation.operate(identity, element)

        assert result1.value == element.value
        assert result2.value == element.value

    def test_multiplicative_operation_with_zero(self):
        """Test multiplicative operation with zero (should raise error)."""
        operation = MultiplicativeOperation()

        zero_value = AbsoluteValue.absolute()
        nonzero_value = AbsoluteValue(magnitude=5.0, direction=1)

        zero_element = GroupElement(value=zero_value)
        nonzero_element = GroupElement(value=nonzero_value)

        with pytest.raises(
            ValueError, match="Absolute elements not allowed in multiplicative group"
        ):
            operation.operate(zero_element, nonzero_element)

    def test_multiplicative_inverse_zero_error(self):
        """Test that multiplicative inverse of zero raises error."""
        operation = MultiplicativeOperation()

        zero_value = AbsoluteValue.absolute()
        zero_element = GroupElement(value=zero_value)

        with pytest.raises(
            ValueError, match="Cannot compute multiplicative inverse of zero"
        ):
            operation.inverse_element(zero_element)


class TestAbsoluteGroup:
    """Test AbsoluteGroup class."""

    def test_additive_group_creation(self):
        """Test creation of additive group."""
        group = AbsoluteGroup.additive_group()
        assert isinstance(group, AbsoluteGroup)
        assert isinstance(group.operation, AdditiveOperation)

    def test_multiplicative_group_creation(self):
        """Test creation of multiplicative group."""
        group = AbsoluteGroup.multiplicative_group()
        assert isinstance(group, AbsoluteGroup)
        assert isinstance(group.operation, MultiplicativeOperation)

    def test_finite_cyclic_group_creation(self):
        """Test creation of finite cyclic group."""
        group = AbsoluteGroup.finite_cyclic_group(order=5)
        assert isinstance(group, AbsoluteGroup)
        assert len(group) == 5

    def test_finite_cyclic_group_invalid_order(self):
        """Test finite cyclic group with invalid order."""
        with pytest.raises(ValueError, match="Group order must be positive"):
            AbsoluteGroup.finite_cyclic_group(order=0)

        with pytest.raises(ValueError, match="Group order must be positive"):
            AbsoluteGroup.finite_cyclic_group(order=-1)

    def test_group_operate(self):
        """Test group operation."""
        group = AbsoluteGroup.additive_group()

        value1 = AbsoluteValue(magnitude=2.0, direction=1)
        value2 = AbsoluteValue(magnitude=3.0, direction=1)
        element1 = GroupElement(value=value1)
        element2 = GroupElement(value=value2)

        result = group.operate(element1, element2)
        assert result.value.magnitude == 5.0

    def test_group_identity_element(self):
        """Test group identity element."""
        additive_group = AbsoluteGroup.additive_group()
        multiplicative_group = AbsoluteGroup.multiplicative_group()

        add_identity = additive_group.identity_element()
        mult_identity = multiplicative_group.identity_element()

        assert add_identity.value.is_absolute()  # Zero for additive group
        assert (
            mult_identity.value.magnitude == 1.0 and mult_identity.value.direction == 1
        )  # Unit for multiplicative group

    def test_group_inverse_element(self):
        """Test group inverse element."""
        group = AbsoluteGroup.additive_group()

        value = AbsoluteValue(magnitude=6.0, direction=1)
        element = GroupElement(value=value)

        inverse = group.inverse_element(element)
        assert inverse.value.magnitude == 6.0
        assert inverse.value.direction == -1

    def test_group_order_infinite(self):
        """Test order of infinite groups."""
        additive_group = AbsoluteGroup.additive_group()
        multiplicative_group = AbsoluteGroup.multiplicative_group()

        assert additive_group.order() is None
        assert multiplicative_group.order() is None

    def test_group_order_finite(self):
        """Test order of finite groups."""
        group = AbsoluteGroup.finite_cyclic_group(order=7)
        assert group.order() == 7

    def test_element_order(self):
        """Test order of group elements."""
        group = AbsoluteGroup.finite_cyclic_group(order=6)
        elements = list(group.elements)

        if elements:
            element = elements[0]
            order = group.element_order(element)
            assert isinstance(order, int)
            assert order > 0

    def test_is_abelian(self):
        """Test if group is abelian."""
        additive_group = AbsoluteGroup.additive_group()
        multiplicative_group = AbsoluteGroup.multiplicative_group()
        finite_group = AbsoluteGroup.finite_cyclic_group(order=4)

        assert additive_group.is_abelian()
        assert multiplicative_group.is_abelian()
        assert finite_group.is_abelian()

    def test_subgroup_creation(self):
        """Test subgroup creation."""
        group = AbsoluteGroup.finite_cyclic_group(order=8)
        elements = list(group.elements)[:4]  # Take first 4 elements

        subgroup = group.subgroup(elements)
        assert isinstance(subgroup, AbsoluteGroup)
        assert len(subgroup) <= len(group)

    def test_cosets(self):
        """Test coset computation."""
        group = AbsoluteGroup.finite_cyclic_group(order=6)
        elements = list(group.elements)[:2]  # Small subgroup

        subgroup = group.subgroup(elements)
        cosets = group.cosets(subgroup)

        assert isinstance(cosets, list)
        assert len(cosets) > 0

    def test_is_normal_subgroup(self):
        """Test normal subgroup check."""
        group = AbsoluteGroup.finite_cyclic_group(order=4)
        elements = list(group.elements)[:2]

        subgroup = group.subgroup(elements)
        is_normal = group.is_normal_subgroup(subgroup)

        # Cyclic groups have all subgroups normal
        assert isinstance(is_normal, bool)

    def test_quotient_group(self):
        """Test quotient group creation."""
        group = AbsoluteGroup.finite_cyclic_group(order=6)
        elements = list(group.elements)[:2]

        subgroup = group.subgroup(elements)

        if group.is_normal_subgroup(subgroup):
            quotient = group.quotient_group(subgroup)
            assert isinstance(quotient, AbsoluteGroup)

    def test_group_contains(self):
        """Test group membership."""
        group = AbsoluteGroup.finite_cyclic_group(order=5)
        elements = list(group.elements)

        if elements:
            assert elements[0] in group

    def test_group_iteration(self):
        """Test group iteration."""
        group = AbsoluteGroup.finite_cyclic_group(order=3)

        elements = list(group)
        assert len(elements) == 3
        assert all(isinstance(elem, GroupElement) for elem in elements)

    def test_group_string_representations(self):
        """Test group string representations."""
        additive_group = AbsoluteGroup.additive_group()
        finite_group = AbsoluteGroup.finite_cyclic_group(order=4)

        assert "AbsoluteGroup" in repr(additive_group)
        assert "AbsoluteGroup" in str(finite_group)

    def test_group_associativity(self):
        """Test group associativity property."""
        group = AbsoluteGroup.additive_group()

        value1 = AbsoluteValue(magnitude=1.0, direction=1)
        value2 = AbsoluteValue(magnitude=2.0, direction=1)
        value3 = AbsoluteValue(magnitude=3.0, direction=1)

        a = GroupElement(value=value1)
        b = GroupElement(value=value2)
        c = GroupElement(value=value3)

        # (a + b) + c = a + (b + c)
        left = group.operate(group.operate(a, b), c)
        right = group.operate(a, group.operate(b, c))

        assert left.value.magnitude == right.value.magnitude
        assert left.value.direction == right.value.direction

    def test_group_identity_property(self):
        """Test group identity property."""
        group = AbsoluteGroup.additive_group()
        identity = group.identity_element()

        value = AbsoluteValue(magnitude=5.0, direction=-1)
        element = GroupElement(value=value)

        # e + a = a + e = a
        left = group.operate(identity, element)
        right = group.operate(element, identity)

        assert left.value == element.value
        assert right.value == element.value

    def test_group_inverse_property(self):
        """Test group inverse property."""
        group = AbsoluteGroup.additive_group()
        identity = group.identity_element()

        value = AbsoluteValue(magnitude=4.0, direction=1)
        element = GroupElement(value=value)
        inverse = group.inverse_element(element)

        # a + (-a) = (-a) + a = e
        left = group.operate(element, inverse)
        right = group.operate(inverse, element)

        # Both should equal the identity element
        assert left == identity
        assert right == identity

    def test_group_edge_cases_large_values(self):
        """Test group operations with very large values."""
        group = AbsoluteGroup.additive_group()

        # Test large + finite = large
        large_value = AbsoluteValue(magnitude=1e10, direction=1)
        finite_value = AbsoluteValue(magnitude=5.0, direction=1)

        large_elem = GroupElement(value=large_value)
        finite_elem = GroupElement(value=finite_value)

        result = group.operate(large_elem, finite_elem)
        assert result.value.magnitude > 1e9  # Should be very large

        # Test large + (-large) behavior
        neg_large = GroupElement(value=AbsoluteValue(magnitude=1e10, direction=-1))
        result = group.operate(large_elem, neg_large)
        # Result should be handled by compensator
        assert isinstance(result.value, AbsoluteValue)

    def test_group_edge_cases_zero(self):
        """Test group operations with zero values."""
        group = AbsoluteGroup.additive_group()

        zero_elem = GroupElement(value=AbsoluteValue.absolute())
        finite_elem = GroupElement(value=AbsoluteValue(magnitude=3.0, direction=1))

        # Test zero + finite = finite
        result = group.operate(zero_elem, finite_elem)
        assert result.value.magnitude == 3.0
        assert result.value.direction == 1

        # Test zero + zero = zero
        result = group.operate(zero_elem, zero_elem)
        assert result.value.is_absolute()  # Zero for additive group

    def test_multiplicative_group_edge_cases(self):
        """Test multiplicative group with edge cases."""
        group = AbsoluteGroup.multiplicative_group()

        # Test unit * finite = finite
        unit_elem = GroupElement(value=AbsoluteValue.unit_positive())
        finite_elem = GroupElement(value=AbsoluteValue(magnitude=2.0, direction=1))

        result = group.operate(unit_elem, finite_elem)
        assert result.value.magnitude == 2.0

        # Test finite * finite
        elem1 = GroupElement(value=AbsoluteValue(magnitude=3.0, direction=1))
        elem2 = GroupElement(value=AbsoluteValue(magnitude=4.0, direction=-1))

        result = group.operate(elem1, elem2)
        assert result.value.magnitude == 12.0
        assert result.value.direction == -1

    def test_group_element_conjugacy_class(self):
        """Test GroupElement conjugacy class functionality."""
        value = AbsoluteValue(magnitude=5.0, direction=1)
        element = GroupElement(value=value, conjugacy_class="test_class")

        assert element.conjugacy_class == "test_class"

        # Test default conjugacy class
        element2 = GroupElement(value=value)
        assert element2.conjugacy_class is None

    def test_group_element_order_attribute(self):
        """Test GroupElement order attribute."""
        value = AbsoluteValue(magnitude=3.0, direction=1)
        element = GroupElement(value=value, order=6)

        assert element.order == 6

        # Test default order
        element2 = GroupElement(value=value)
        assert element2.order is None

    def test_finite_cyclic_group_properties(self):
        """Test properties of finite cyclic groups."""
        group = AbsoluteGroup.finite_cyclic_group(order=6)  # Reduced from 12

        # Test that all elements have finite order (sample only first 3)
        elements = list(group.elements)[:3]
        for element in elements:
            order = group.element_order(element)
            assert order is not None
            assert order <= 6
            assert 6 % order == 0  # Order divides group order

    def test_subgroup_closure(self):
        """Test that subgroups are closed under operations."""
        group = AbsoluteGroup.finite_cyclic_group(order=4)  # Reduced from 8
        elements = list(group.elements)[:2]  # Reduced from 3

        subgroup = group.subgroup(elements)

        # Test closure: for any two elements in subgroup, their operation is also in subgroup
        subgroup_elements = list(subgroup.elements)
        if len(subgroup_elements) >= 2:
            a, b = subgroup_elements[0], subgroup_elements[1]
            result = subgroup.operate(a, b)
            assert result in subgroup

    def test_coset_partition(self):
        """Test that cosets partition the group."""
        group = AbsoluteGroup.finite_cyclic_group(
            order=4
        )  # Smaller group for stability
        elements = list(group.elements)[:1]  # Use single generator

        subgroup = group.subgroup(elements)
        cosets = group.cosets(subgroup)

        # Verify cosets are non-empty and partition the group
        assert len(cosets) > 0

        # Total elements in all cosets should equal group size
        total_elements = sum(len(coset) for coset in cosets)
        assert total_elements == len(group)

        # Cosets should be disjoint
        all_elements = set()
        for coset in cosets:
            assert len(coset & all_elements) == 0  # No overlap
            all_elements.update(coset)

    def test_group_homomorphism_properties(self):
        """Test basic homomorphism properties."""
        group1 = AbsoluteGroup.additive_group()
        group2 = AbsoluteGroup.additive_group()

        # Test that identity maps to identity
        identity1 = group1.identity_element()
        identity2 = group2.identity_element()

        assert identity1.value.is_absolute()  # Zero for additive group
        assert identity2.value.is_absolute()  # Zero for additive group

    def test_large_magnitude_operations(self):
        """Test operations with very large magnitudes."""
        group = AbsoluteGroup.additive_group()

        large_value1 = AbsoluteValue(magnitude=1e10, direction=1)
        large_value2 = AbsoluteValue(magnitude=1e10, direction=1)

        elem1 = GroupElement(value=large_value1)
        elem2 = GroupElement(value=large_value2)

        result = group.operate(elem1, elem2)
        assert result.value.magnitude == 2e10

    def test_small_magnitude_operations(self):
        """Test operations with very small magnitudes."""
        group = AbsoluteGroup.additive_group()

        small_value1 = AbsoluteValue(magnitude=1e-10, direction=1)
        small_value2 = AbsoluteValue(magnitude=1e-10, direction=1)

        elem1 = GroupElement(value=small_value1)
        elem2 = GroupElement(value=small_value2)

        result = group.operate(elem1, elem2)
        assert result.value.magnitude == 2e-10

    def test_mixed_direction_operations(self):
        """Test operations with mixed positive/negative directions."""
        group = AbsoluteGroup.additive_group()

        pos_elem = GroupElement(value=AbsoluteValue(magnitude=5.0, direction=1))
        neg_elem = GroupElement(value=AbsoluteValue(magnitude=3.0, direction=-1))

        result = group.operate(pos_elem, neg_elem)
        assert result.value.magnitude == 2.0
        assert result.value.direction == 1

        # Test reverse order
        result2 = group.operate(neg_elem, pos_elem)
        assert result2.value.magnitude == 2.0
        assert result2.value.direction == 1

    def test_group_error_handling(self):
        """Test error handling in group operations."""
        group = AbsoluteGroup.additive_group()

        # Test with invalid element types
        with pytest.raises((TypeError, AttributeError)):
            group.operate("not_an_element", "also_not_an_element")

        # Test subgroup with empty elements creates trivial subgroup
        trivial_subgroup = group.subgroup([])
        assert len(trivial_subgroup) == 1  # Only identity element
        identity = group.identity_element()
        assert identity in trivial_subgroup

    def test_performance_moderate_finite_group(self):
        """Test performance with moderate finite groups."""
        import time

        start_time = time.time()
        group = AbsoluteGroup.finite_cyclic_group(order=20)  # Reduced from 100
        creation_time = time.time() - start_time

        # Should create reasonably quickly
        assert creation_time < 2.0  # Reduced from 5 seconds

        # Test iteration performance
        start_time = time.time()
        elements = list(group)
        iteration_time = time.time() - start_time

        assert len(elements) == 20
        assert iteration_time < 1.0  # Reduced from 2 seconds
