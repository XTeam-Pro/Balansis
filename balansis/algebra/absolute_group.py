# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.
#
# This file is part of Balansis.
# Balansis is dual-licensed under:
#   1. GNU Affero General Public License v3.0 (AGPLv3) for open-source use.
#   2. A Commercial License for proprietary and corporate use.
#
# See LICENSING.md in the project root for license selection details.
# For commercial licensing: andrew@xteam.pro
"""AbsoluteGroup implementation for Balansis library.

This module implements the AbsoluteGroup algebraic structure, which provides
group operations for AbsoluteValue elements according to Absolute Compensation
Theory (ACT) principles. The group maintains closure, associativity, identity,
and inverse properties while handling Absolute elements appropriately.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core.absolute import AbsoluteValue
from ..logic.compensator import Compensator


class GroupElement(BaseModel):
    """Wrapper for AbsoluteValue elements in the group context.

    Provides additional group-specific properties and methods while
    maintaining the underlying AbsoluteValue semantics.

    Attributes:
        value: The underlying AbsoluteValue
        order: Order of the element in the group (if finite)
        conjugacy_class: Conjugacy class identifier
    """

    value: AbsoluteValue
    order: Optional[int] = Field(default=None, description="Order of element in group")
    conjugacy_class: Optional[str] = Field(
        default=None, description="Conjugacy class ID"
    )

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("order")
    @classmethod
    def validate_order(cls, v: Optional[int]) -> Optional[int]:
        """Ensure order is positive if specified."""
        if v is not None and v <= 0:
            raise ValueError("Element order must be positive")
        return v

    def __hash__(self) -> int:
        """Hash based on underlying AbsoluteValue."""
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        """Equality based on underlying AbsoluteValue."""
        if not isinstance(other, GroupElement):
            return False
        return self.value == other.value

    def __repr__(self) -> str:
        """String representation for debugging."""
        order_str = f", order={self.order}" if self.order else ""
        return f"GroupElement({self.value}{order_str})"


class GroupOperation(ABC):
    """Abstract base class for group operations."""

    @abstractmethod
    def apply(self, a: GroupElement, b: GroupElement) -> GroupElement:
        """Apply the group operation to two elements.

        Args:
            a: First group element
            b: Second group element

        Returns:
            Result of the group operation
        """
        pass

    @abstractmethod
    def identity(self) -> GroupElement:
        """Return the identity element for this operation.

        Returns:
            Identity element
        """
        pass

    @abstractmethod
    def inverse(self, element: GroupElement) -> GroupElement:
        """Return the inverse of an element.

        Args:
            element: Group element to invert

        Returns:
            Inverse element
        """
        pass


class AdditiveOperation(GroupOperation):
    """Additive group operation for AbsoluteValues.

    Implements the additive group structure where:
    - Operation: compensated addition
    - Identity: Absolute (zero equivalent)
    - Inverse: negation
    """

    def __init__(self, compensator: Optional[Compensator] = None):
        """Initialize additive operation.

        Args:
            compensator: Compensator for stable operations
        """
        self.compensator = compensator or Compensator()

    def apply(self, a: GroupElement, b: GroupElement) -> GroupElement:
        """Apply addition operation.

        Args:
            a: First group element
            b: Second group element

        Returns:
            Sum as GroupElement
        """
        # Use direct addition for group operations to ensure proper identity behavior
        result_value = a.value + b.value
        return GroupElement(value=result_value)

    def identity(self) -> GroupElement:
        """Return additive identity (Absolute).

        Returns:
            Absolute as GroupElement
        """
        return GroupElement(value=AbsoluteValue.absolute(), order=1)

    def inverse(self, element: GroupElement) -> GroupElement:
        """Return additive inverse (negation).

        Args:
            element: Group element to invert

        Returns:
            Negated element
        """
        inverse_value = -element.value
        return GroupElement(value=inverse_value)

    def operate(self, a: GroupElement, b: GroupElement) -> GroupElement:
        """Apply additive operation (alias for apply).

        Args:
            a: First group element
            b: Second group element

        Returns:
            Sum as GroupElement
        """
        return self.apply(a, b)

    def identity_element(self) -> GroupElement:
        """Return additive identity element (alias for identity).

        Returns:
            Absolute as GroupElement
        """
        return self.identity()

    def inverse_element(self, element: GroupElement) -> GroupElement:
        """Return additive inverse element (alias for inverse).

        Args:
            element: Group element to invert

        Returns:
            Negated element
        """
        return self.inverse(element)


class MultiplicativeOperation(GroupOperation):
    """Multiplicative group operation for non-Absolute AbsoluteValues.

    Implements the multiplicative group structure where:
    - Operation: compensated multiplication
    - Identity: unit positive AbsoluteValue
    - Inverse: reciprocal (via EternalRatio)

    Note: Absolute elements are excluded from this group as they
    don't have multiplicative inverses.
    """

    def __init__(self, compensator: Optional[Compensator] = None):
        """Initialize multiplicative operation.

        Args:
            compensator: Compensator for stable operations
        """
        self.compensator = compensator or Compensator()

    def apply(self, a: GroupElement, b: GroupElement) -> GroupElement:
        """Apply compensated multiplication.

        Args:
            a: First group element
            b: Second group element

        Returns:
            Product as GroupElement

        Raises:
            ValueError: If either element is Absolute
        """
        if a.value.is_absolute() or b.value.is_absolute():
            raise ValueError("Absolute elements not allowed in multiplicative group")

        result_value = self.compensator.compensate_multiplication(a.value, b.value)
        return GroupElement(value=result_value)

    def identity(self) -> GroupElement:
        """Return multiplicative identity (unit positive).

        Returns:
            Unit positive as GroupElement
        """
        return GroupElement(value=AbsoluteValue.unit_positive(), order=1)

    def inverse(self, element: GroupElement) -> GroupElement:
        """Return multiplicative inverse.

        Args:
            element: Group element to invert

        Returns:
            Inverse element

        Raises:
            ValueError: If element is Absolute
        """
        if element.value.is_absolute():
            raise ValueError("Absolute elements have no multiplicative inverse")

        inverse_value = element.value.inverse()
        return GroupElement(value=inverse_value)

    def operate(self, a: GroupElement, b: GroupElement) -> GroupElement:
        """Apply multiplicative operation (alias for apply).

        Args:
            a: First group element
            b: Second group element

        Returns:
            Product as GroupElement
        """
        return self.apply(a, b)

    def identity_element(self) -> GroupElement:
        """Return multiplicative identity element (alias for identity).

        Returns:
            Unit positive as GroupElement
        """
        return self.identity()

    def inverse_element(self, element: GroupElement) -> GroupElement:
        """Return multiplicative inverse element (alias for inverse).

        Args:
            element: Group element to invert

        Returns:
            Inverse element
        """
        if element.value.is_absolute():
            raise ValueError("Cannot compute multiplicative inverse of zero")
        return self.inverse(element)


class CyclicOperation(AdditiveOperation):
    """Addition of integer residues modulo a positive order."""

    def __init__(self, modulus: int):
        super().__init__()
        self.modulus = modulus

    def apply(self, a: GroupElement, b: GroupElement) -> GroupElement:
        value = (int(a.value.to_float()) + int(b.value.to_float())) % self.modulus
        return GroupElement(value=AbsoluteValue.from_float(float(value)))

    def inverse(self, element: GroupElement) -> GroupElement:
        value = -int(element.value.to_float()) % self.modulus
        return GroupElement(value=AbsoluteValue.from_float(float(value)))


class AbsoluteGroup:
    """Group structure for AbsoluteValue elements.

    Provides a complete group implementation with support for both
    additive and multiplicative operations, subgroup analysis,
    and group-theoretic properties.

    Attributes:
        operation: The group operation (additive or multiplicative)
        elements: Set of group elements
        compensator: Compensator for stable operations
        finite: Whether the group is finite

    Examples:
        >>> # Create additive group
        >>> group = AbsoluteGroup.additive_group()
        >>> a = GroupElement(value=AbsoluteValue(2.0, 1))
        >>> b = GroupElement(value=AbsoluteValue(3.0, -1))
        >>> result = group.operate(a, b)

        >>> # Create multiplicative group
        >>> mult_group = AbsoluteGroup.multiplicative_group()
        >>> x = GroupElement(value=AbsoluteValue(2.0, 1))
        >>> y = GroupElement(value=AbsoluteValue(0.5, 1))
        >>> product = mult_group.operate(x, y)
    """

    def __init__(
        self,
        operation: GroupOperation,
        elements: Optional[Set[GroupElement]] = None,
        finite: bool = False,
    ):
        """Initialize AbsoluteGroup.

        Args:
            operation: Group operation to use
            elements: Set of group elements (None for infinite groups)
            finite: Whether the group is finite
        """
        self.operation = operation
        self.elements = elements or set()
        self.finite = finite
        self._identity: Optional[GroupElement] = None
        self._order: Optional[int] = None

    @classmethod
    def additive_group(
        cls, compensator: Optional[Compensator] = None
    ) -> "AbsoluteGroup":
        """Create an additive group of AbsoluteValues.

        Args:
            compensator: Optional compensator for operations

        Returns:
            AbsoluteGroup with additive operation
        """
        operation = AdditiveOperation(compensator)
        return cls(operation=operation, finite=False)

    @classmethod
    def multiplicative_group(
        cls, compensator: Optional[Compensator] = None
    ) -> "AbsoluteGroup":
        """Create a multiplicative group of non-Absolute AbsoluteValues.

        Args:
            compensator: Optional compensator for operations

        Returns:
            AbsoluteGroup with multiplicative operation
        """
        operation = MultiplicativeOperation(compensator)
        return cls(operation=operation, finite=False)

    @classmethod
    def finite_cyclic_group(
        cls, order: int, compensator: Optional[Compensator] = None
    ) -> "AbsoluteGroup":
        """Construct the additive group of residues modulo ``order``."""
        if not isinstance(order, int) or isinstance(order, bool) or order <= 0:
            raise ValueError("Group order must be positive")
        elements = {
            GroupElement(
                value=AbsoluteValue.from_float(float(i)),
                order=order // math.gcd(i, order),
            )
            for i in range(order)
        }
        group = cls(operation=CyclicOperation(order), elements=elements, finite=True)
        group._order = order
        return group

    def operate(self, a: GroupElement, b: GroupElement) -> GroupElement:
        """Apply the group operation to two elements.

        Args:
            a: First group element
            b: Second group element

        Returns:
            Result of group operation

        Raises:
            ValueError: If elements not in finite group
        """
        if self.finite and (a not in self.elements or b not in self.elements):
            raise ValueError("Elements must be in the group")

        result = self.operation.apply(a, b)

        # Add result to finite group if not present
        if self.finite:
            if result not in self.elements:
                raise ValueError("Finite group is not closed under its operation")

        return result

    def identity_element(self) -> GroupElement:
        """Get the identity element of the group.

        Returns:
            Identity element
        """
        if self._identity is None:
            self._identity = self.operation.identity()
            if self.finite:
                self.elements.add(self._identity)

        return self._identity

    def inverse_element(self, element: GroupElement) -> GroupElement:
        """Get the inverse of an element.

        Args:
            element: Element to invert

        Returns:
            Inverse element

        Raises:
            ValueError: If element not in finite group
        """
        if self.finite and element not in self.elements:
            raise ValueError("Element must be in the group")

        inverse = self.operation.inverse(element)

        if self.finite:
            if inverse not in self.elements:
                raise ValueError("Inverse is not in finite group")

        return inverse

    def order(self) -> Optional[int]:
        """Get the order of the group.

        Returns:
            Order of the group (None if infinite)
        """
        if self.finite:
            return self._order or len(self.elements)
        return None

    def element_order(self, element: GroupElement) -> Optional[int]:
        """Calculate the order of an element.

        Args:
            element: Element to find order of

        Returns:
            Order of the element (None if infinite)
        """
        if element.order is not None:
            return element.order

        # For finite groups, order divides group order
        if self.finite and self._order:
            max_iterations = min(self._order, 100)  # Limit to reasonable size
        else:
            max_iterations = 50  # Much smaller limit for infinite groups

        # Calculate order by repeated application
        identity = self.identity_element()
        current = element
        order = 1

        # Check if element is already identity
        if self._elements_equal(current, identity):
            element.order = 1
            return 1

        while order < max_iterations:
            current = self.operate(current, element)
            order += 1

            if self._elements_equal(current, identity):
                element.order = order
                return order

        # Element has infinite order or very large finite order
        return None

    def _elements_equal(self, elem1: GroupElement, elem2: GroupElement) -> bool:
        """Check if two group elements are equal.

        Args:
            elem1: First element
            elem2: Second element

        Returns:
            True if elements are equal
        """
        # Compare the underlying values
        return elem1.value == elem2.value

    def is_abelian(self) -> bool:
        """Check if the group is abelian (commutative).

        Returns:
            True if group is abelian
        """
        if not self.finite:
            # For infinite groups, check operation type
            return isinstance(
                self.operation, (AdditiveOperation, MultiplicativeOperation)
            )

        # Check commutativity for all pairs in finite group
        elements_list = list(self.elements)
        for i, a in enumerate(elements_list):
            for j, b in enumerate(elements_list[i:], i):
                if self.operate(a, b) != self.operate(b, a):
                    return False

        return True

    def subgroup(self, generators: List[GroupElement]) -> "AbsoluteGroup":
        """Compute finite closure, rejecting an unbounded generated subgroup."""
        if self.finite and any(g not in self.elements for g in generators):
            raise ValueError("Generators must belong to the group")
        elements = {self.identity_element()}
        steps = list(generators) + [self.inverse_element(g) for g in generators]
        queue = list(elements)
        while queue:
            current = queue.pop()
            for step in steps:
                result = self.operation.apply(current, step)
                if result not in elements:
                    if len(elements) >= (len(self.elements) if self.finite else 1000):
                        raise ValueError(
                            "Generated subgroup exceeds finite closure bound"
                        )
                    elements.add(result)
                    queue.append(result)
        subgroup = AbsoluteGroup(
            operation=self.operation, elements=elements, finite=True
        )
        subgroup._order = len(elements)
        return subgroup

    def cosets(
        self, subgroup: "AbsoluteGroup", left: bool = True
    ) -> List[Set[GroupElement]]:
        """Compute left or right cosets of a subgroup.

        Args:
            subgroup: Subgroup to compute cosets for
            left: If True, compute left cosets; otherwise right cosets

        Returns:
            List of cosets (each coset is a set of elements)

        Raises:
            ValueError: If groups are not finite
        """
        if not (self.finite and subgroup.finite):
            raise ValueError("Coset computation requires finite groups")

        cosets = []
        remaining_elements = set(self.elements)

        while remaining_elements:
            representative = next(iter(remaining_elements))
            coset = set()

            for h in subgroup.elements:
                if left:
                    coset_element = self.operate(representative, h)
                else:
                    coset_element = self.operate(h, representative)
                coset.add(coset_element)

            cosets.append(coset)
            remaining_elements -= coset

        return cosets

    def is_normal_subgroup(self, subgroup: "AbsoluteGroup") -> bool:
        """Check if a subgroup is normal.

        Args:
            subgroup: Subgroup to check

        Returns:
            True if subgroup is normal
        """
        if not self.finite:
            # For infinite groups, assume abelian groups have all normal subgroups
            return self.is_abelian()

        # Check if gHg^(-1) = H for all g in G
        for g in list(self.elements):
            g_inv = self.inverse_element(g)

            for h in subgroup.elements:
                # Compute ghg^(-1)
                temp = self.operate(g, h)
                conjugate = self.operate(temp, g_inv)

                if conjugate not in subgroup.elements:
                    return False

        return True

    def quotient_group(self, normal_subgroup: "AbsoluteGroup") -> "AbsoluteGroup":
        """Construct a quotient of a finite cyclic group."""
        if not (self.finite and normal_subgroup.finite):
            raise ValueError("Quotient group computation requires finite groups")
        if not self.is_normal_subgroup(normal_subgroup):
            raise ValueError("Subgroup must be normal for quotient group")
        if isinstance(self.operation, CyclicOperation):
            return AbsoluteGroup.finite_cyclic_group(
                len(self.elements) // len(normal_subgroup.elements)
            )
        raise NotImplementedError("Quotient operations require a finite cyclic group")

    def __len__(self) -> int:
        """Return the number of elements (for finite groups).

        Returns:
            Number of elements

        Raises:
            ValueError: If group is infinite
        """
        if not self.finite:
            raise ValueError("Infinite groups have no finite length")
        return len(self.elements)

    def __contains__(self, element: GroupElement) -> bool:
        """Check if element is in the group.

        Args:
            element: Element to check

        Returns:
            True if element is in group
        """
        if self.finite:
            return element in self.elements

        # For infinite groups, check if element is valid for the operation
        if isinstance(self.operation, MultiplicativeOperation):
            return not element.value.is_absolute()

        return True  # All AbsoluteValues valid for additive group

    def __iter__(self) -> Iterator[GroupElement]:
        """Iterate over group elements (finite groups only).

        Returns:
            Iterator over group elements

        Raises:
            ValueError: If group is infinite
        """
        if not self.finite:
            raise ValueError("Cannot iterate over infinite group")
        return iter(self.elements)

    def __repr__(self) -> str:
        """String representation for debugging."""
        op_name = type(self.operation).__name__
        if self.finite:
            return f"AbsoluteGroup({op_name}, order={len(self.elements)})"
        return f"AbsoluteGroup({op_name}, infinite)"

    def __str__(self) -> str:
        """Human-readable string representation."""
        op_type = (
            "additive"
            if isinstance(self.operation, AdditiveOperation)
            else "multiplicative"
        )
        if self.finite:
            return f"Finite {op_type} AbsoluteGroup of order {len(self.elements)}"
        return f"Infinite {op_type} AbsoluteGroup"
