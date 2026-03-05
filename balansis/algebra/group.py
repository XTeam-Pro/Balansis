# Copyright (c) 2024-2026 Tikhonov Andrey. All rights reserved.
# SPDX-License-Identifier: MIT (non-commercial) | Commercial use: see COMMERCIAL_LICENSE.md
"""AbsoluteGroup implementation for Balansis library.

This module implements the AbsoluteGroup algebraic structure, which provides
group operations for AbsoluteValue elements according to Absolute Compensation
Theory (ACT) principles. The group maintains closure, associativity, identity,
and inverse properties while handling Absolute elements appropriately.
"""

from typing import List, Set, Optional, Iterator, Tuple
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, validator
import math

from ..core.absolute import AbsoluteValue
from ..logic.compensator import Compensator, CompensationStrategy


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
    conjugacy_class: Optional[str] = Field(default=None, description="Conjugacy class ID")
    
    @validator('order')
    def validate_order(cls, v: Optional[int]) -> Optional[int]:
        """Ensure order is positive if specified."""
        if v is not None and v <= 0:
            raise ValueError('Element order must be positive')
        return v
    
    def __hash__(self) -> int:
        """Hash based on underlying AbsoluteValue."""
        return hash(self.value)
    
    def __eq__(self, other) -> bool:
        """Equality based on underlying AbsoluteValue."""
        if not isinstance(other, GroupElement):
            return False
        return self.value == other.value
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        order_str = f", order={self.order}" if self.order else ""
        return f"GroupElement({self.value}{order_str})"
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True


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
        """Apply compensated addition.
        
        Args:
            a: First group element
            b: Second group element
            
        Returns:
            Sum as GroupElement
        """
        result_value = self.compensator.compensate_addition(a.value, b.value)
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
    
    def __init__(self, operation: GroupOperation, 
                 elements: Optional[Set[GroupElement]] = None,
                 finite: bool = False):
        """Initialize AbsoluteGroup.
        
        Args:
            operation: Group operation to use
            elements: Set of group elements (None for infinite groups)
            finite: Whether the group is finite
        """
        self.operation = operation
        self.elements = elements or set()
        self.finite = finite
        self._identity = None
        self._order = None
    
    @classmethod
    def additive_group(cls, compensator: Optional[Compensator] = None) -> 'AbsoluteGroup':
        """Create an additive group of AbsoluteValues.
        
        Args:
            compensator: Optional compensator for operations
            
        Returns:
            AbsoluteGroup with additive operation
        """
        operation = AdditiveOperation(compensator)
        return cls(operation=operation, finite=False)
    
    @classmethod
    def multiplicative_group(cls, compensator: Optional[Compensator] = None) -> 'AbsoluteGroup':
        """Create a multiplicative group of non-Absolute AbsoluteValues.
        
        Args:
            compensator: Optional compensator for operations
            
        Returns:
            AbsoluteGroup with multiplicative operation
        """
        operation = MultiplicativeOperation(compensator)
        return cls(operation=operation, finite=False)
    
    @classmethod
    def finite_cyclic_group(cls, order: int, 
                          compensator: Optional[Compensator] = None) -> 'AbsoluteGroup':
        """Create a finite cyclic group.
        
        Args:
            order: Order of the cyclic group
            compensator: Optional compensator for operations
            
        Returns:
            Finite cyclic AbsoluteGroup
        """
        if order <= 0:
            raise ValueError("Group order must be positive")
        
        operation = AdditiveOperation(compensator)
        
        # Generate cyclic group elements
        elements = set()
        generator = AbsoluteValue(magnitude=2 * math.pi / order, direction=1)
        
        for i in range(order):
            element_value = AbsoluteValue(
                magnitude=generator.magnitude * i,
                direction=1 if i % 2 == 0 else -1
            )
            elements.add(GroupElement(value=element_value, order=order))
        
        group = cls(operation=operation, elements=elements, finite=True)
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
            self.elements.add(result)
        
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
            self.elements.add(inverse)
        
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
        
        # Calculate order by repeated application
        identity = self.identity_element()
        current = element
        order = 1
        max_iterations = 1000  # Prevent infinite loops
        
        while order <= max_iterations:
            if current == identity:
                element.order = order
                return order
            
            current = self.operate(current, element)
            order += 1
        
        # Element has infinite order or very large finite order
        return None
    
    def is_abelian(self) -> bool:
        """Check if the group is abelian (commutative).
        
        Returns:
            True if group is abelian
        """
        if not self.finite:
            # For infinite groups, check operation type
            return isinstance(self.operation, (AdditiveOperation, MultiplicativeOperation))
        
        # Check commutativity for all pairs in finite group
        elements_list = list(self.elements)
        for i, a in enumerate(elements_list):
            for j, b in enumerate(elements_list[i:], i):
                if self.operate(a, b) != self.operate(b, a):
                    return False
        
        return True
    
    def subgroup(self, generators: List[GroupElement]) -> 'AbsoluteGroup':
        """Generate a subgroup from a set of generators.
        
        Args:
            generators: List of generating elements
            
        Returns:
            Subgroup generated by the elements
        """
        if not generators:
            # Trivial subgroup
            identity = self.identity_element()
            return AbsoluteGroup(
                operation=self.operation,
                elements={identity},
                finite=True
            )
        
        # Generate subgroup elements
        subgroup_elements = {self.identity_element()}
        queue = list(generators)
        
        while queue:
            current = queue.pop(0)
            if current not in subgroup_elements:
                subgroup_elements.add(current)
                
                # Add products with existing elements
                for existing in list(subgroup_elements):
                    new_element1 = self.operate(current, existing)
                    new_element2 = self.operate(existing, current)
                    
                    if new_element1 not in subgroup_elements:
                        queue.append(new_element1)
                    if new_element2 not in subgroup_elements:
                        queue.append(new_element2)
                
                # Add inverse
                inverse = self.inverse_element(current)
                if inverse not in subgroup_elements:
                    queue.append(inverse)
        
        subgroup = AbsoluteGroup(
            operation=self.operation,
            elements=subgroup_elements,
            finite=True
        )
        subgroup._order = len(subgroup_elements)
        return subgroup
    
    def cosets(self, subgroup: 'AbsoluteGroup', left: bool = True) -> List[Set[GroupElement]]:
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
    
    def is_normal_subgroup(self, subgroup: 'AbsoluteGroup') -> bool:
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
        for g in self.elements:
            g_inv = self.inverse_element(g)
            
            for h in subgroup.elements:
                # Compute ghg^(-1)
                temp = self.operate(g, h)
                conjugate = self.operate(temp, g_inv)
                
                if conjugate not in subgroup.elements:
                    return False
        
        return True
    
    def quotient_group(self, normal_subgroup: 'AbsoluteGroup') -> 'AbsoluteGroup':
        """Construct quotient group G/N.
        
        Args:
            normal_subgroup: Normal subgroup N
            
        Returns:
            Quotient group G/N
            
        Raises:
            ValueError: If subgroup is not normal or groups not finite
        """
        if not self.is_normal_subgroup(normal_subgroup):
            raise ValueError("Subgroup must be normal for quotient group")
        
        if not (self.finite and normal_subgroup.finite):
            raise ValueError("Quotient group computation requires finite groups")
        
        # Compute left cosets (which equal right cosets for normal subgroups)
        cosets = self.cosets(normal_subgroup, left=True)
        
        # Create quotient group elements (each coset becomes an element)
        quotient_elements = set()
        coset_representatives = {}
        
        for i, coset in enumerate(cosets):
            # Use first element as representative
            representative = next(iter(coset))
            coset_id = f"coset_{i}"
            
            # Create a special GroupElement for the coset
            quotient_element = GroupElement(
                value=representative.value,
                conjugacy_class=coset_id
            )
            quotient_elements.add(quotient_element)
            coset_representatives[quotient_element] = coset
        
        # Create quotient group with inherited operation
        quotient_group = AbsoluteGroup(
            operation=self.operation,
            elements=quotient_elements,
            finite=True
        )
        quotient_group._order = len(quotient_elements)
        
        return quotient_group
    
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
        op_type = "additive" if isinstance(self.operation, AdditiveOperation) else "multiplicative"
        if self.finite:
            return f"Finite {op_type} AbsoluteGroup of order {len(self.elements)}"
        return f"Infinite {op_type} AbsoluteGroup"