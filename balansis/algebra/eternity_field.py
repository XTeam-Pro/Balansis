# Copyright (c) 2024-2026 Tikhonov Andrey. All rights reserved.
# SPDX-License-Identifier: MIT (non-commercial) | Commercial use: see COMMERCIAL_LICENSE.md
"""EternityField implementation for Balansis library.

This module implements the EternityField algebraic structure, which provides
field operations for EternalRatio elements according to Absolute Compensation
Theory (ACT) principles. The field maintains closure, associativity, commutativity,
distributivity, and inverse properties for both addition and multiplication.
"""

from typing import List, Set, Optional, Iterator, Tuple, Union
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, validator
import math
from fractions import Fraction

from ..core.absolute import AbsoluteValue
from ..core.eternity import EternalRatio
from ..logic.compensator import Compensator, CompensationStrategy
from .group import GroupElement, AbsoluteGroup


class FieldElement(BaseModel):
    """Wrapper for EternalRatio elements in the field context.
    
    Provides additional field-specific properties and methods while
    maintaining the underlying EternalRatio semantics.
    
    Attributes:
        ratio: The underlying EternalRatio
        additive_order: Order in the additive group (if finite)
        multiplicative_order: Order in the multiplicative group (if finite)
        minimal_polynomial: Coefficients of minimal polynomial (if algebraic)
    """
    
    ratio: EternalRatio
    additive_order: Optional[int] = Field(default=None, description="Order in additive group")
    multiplicative_order: Optional[int] = Field(default=None, description="Order in multiplicative group")
    minimal_polynomial: Optional[List[float]] = Field(default=None, description="Minimal polynomial coefficients")
    
    @validator('additive_order', 'multiplicative_order')
    def validate_orders(cls, v: Optional[int]) -> Optional[int]:
        """Ensure orders are positive if specified."""
        if v is not None and v <= 0:
            raise ValueError('Element orders must be positive')
        return v
    
    def __hash__(self) -> int:
        """Hash based on underlying EternalRatio."""
        return hash(self.ratio)
    
    def __eq__(self, other) -> bool:
        """Equality based on underlying EternalRatio."""
        if not isinstance(other, FieldElement):
            return False
        return self.ratio == other.ratio
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"FieldElement({self.ratio})"
    
    def is_zero(self) -> bool:
        """Check if this is the additive identity (zero).
        
        Returns:
            True if this represents zero in the field
        """
        return (self.ratio.numerator.is_absolute() and 
                not self.ratio.denominator.is_absolute())
    
    def is_one(self) -> bool:
        """Check if this is the multiplicative identity (one).
        
        Returns:
            True if this represents one in the field
        """
        return self.ratio.numerical_value() == 1.0
    
    def is_unit(self) -> bool:
        """Check if this element has a multiplicative inverse.
        
        Returns:
            True if element is a unit (invertible)
        """
        return not self.is_zero() and self.ratio.is_stable()
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True


class FieldOperation(ABC):
    """Abstract base class for field operations."""
    
    @abstractmethod
    def add(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """Add two field elements.
        
        Args:
            a: First field element
            b: Second field element
            
        Returns:
            Sum of the elements
        """
        pass
    
    @abstractmethod
    def multiply(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """Multiply two field elements.
        
        Args:
            a: First field element
            b: Second field element
            
        Returns:
            Product of the elements
        """
        pass
    
    @abstractmethod
    def additive_identity(self) -> FieldElement:
        """Return the additive identity (zero).
        
        Returns:
            Zero element
        """
        pass
    
    @abstractmethod
    def multiplicative_identity(self) -> FieldElement:
        """Return the multiplicative identity (one).
        
        Returns:
            One element
        """
        pass
    
    @abstractmethod
    def additive_inverse(self, element: FieldElement) -> FieldElement:
        """Return the additive inverse of an element.
        
        Args:
            element: Field element to invert
            
        Returns:
            Additive inverse
        """
        pass
    
    @abstractmethod
    def multiplicative_inverse(self, element: FieldElement) -> FieldElement:
        """Return the multiplicative inverse of an element.
        
        Args:
            element: Field element to invert
            
        Returns:
            Multiplicative inverse
            
        Raises:
            ValueError: If element is zero
        """
        pass


class EternalRatioOperation(FieldOperation):
    """Field operations for EternalRatio elements.
    
    Implements the complete field structure for EternalRatio elements,
    providing both additive and multiplicative operations with proper
    identities and inverses.
    """
    
    def __init__(self, compensator: Optional[Compensator] = None):
        """Initialize field operations.
        
        Args:
            compensator: Compensator for stable operations
        """
        self.compensator = compensator or Compensator()
    
    def add(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """Add two EternalRatio elements.
        
        Implements: (a/b) + (c/d) = (ad + bc)/(bd)
        
        Args:
            a: First field element
            b: Second field element
            
        Returns:
            Sum of the ratios
        """
        # Extract ratios
        ratio_a, ratio_b = a.ratio, b.ratio
        
        # Compute cross products for addition
        # (a/b) + (c/d) = (ad + bc)/(bd)
        ad = self.compensator.compensate_multiplication(ratio_a.numerator, ratio_b.denominator)
        bc = self.compensator.compensate_multiplication(ratio_b.numerator, ratio_a.denominator)
        bd = self.compensator.compensate_multiplication(ratio_a.denominator, ratio_b.denominator)
        
        # Add numerators
        numerator = self.compensator.compensate_addition(ad, bc)
        
        # Create result ratio
        result_ratio = EternalRatio(numerator=numerator, denominator=bd)
        return FieldElement(ratio=result_ratio)
    
    def multiply(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """Multiply two EternalRatio elements.
        
        Implements: (a/b) * (c/d) = (ac)/(bd)
        
        Args:
            a: First field element
            b: Second field element
            
        Returns:
            Product of the ratios
        """
        # Extract ratios
        ratio_a, ratio_b = a.ratio, b.ratio
        
        # Multiply numerators and denominators
        # (a/b) * (c/d) = (ac)/(bd)
        ac = self.compensator.compensate_multiplication(ratio_a.numerator, ratio_b.numerator)
        bd = self.compensator.compensate_multiplication(ratio_a.denominator, ratio_b.denominator)
        
        # Create result ratio
        result_ratio = EternalRatio(numerator=ac, denominator=bd)
        return FieldElement(ratio=result_ratio)
    
    def additive_identity(self) -> FieldElement:
        """Return the additive identity (zero).
        
        Zero in the field is represented as Absolute/Unit = 0/1
        
        Returns:
            Zero element
        """
        zero_ratio = EternalRatio(
            numerator=AbsoluteValue.absolute(),
            denominator=AbsoluteValue.unit_positive()
        )
        return FieldElement(ratio=zero_ratio, additive_order=1)
    
    def multiplicative_identity(self) -> FieldElement:
        """Return the multiplicative identity (one).
        
        One in the field is represented as Unit/Unit = 1/1
        
        Returns:
            One element
        """
        one_ratio = EternalRatio(
            numerator=AbsoluteValue.unit_positive(),
            denominator=AbsoluteValue.unit_positive()
        )
        return FieldElement(ratio=one_ratio, multiplicative_order=1)
    
    def additive_inverse(self, element: FieldElement) -> FieldElement:
        """Return the additive inverse (negation).
        
        Implements: -(a/b) = (-a)/b
        
        Args:
            element: Field element to invert
            
        Returns:
            Additive inverse
        """
        ratio = element.ratio
        negated_numerator = -ratio.numerator
        
        inverse_ratio = EternalRatio(
            numerator=negated_numerator,
            denominator=ratio.denominator
        )
        return FieldElement(ratio=inverse_ratio)
    
    def multiplicative_inverse(self, element: FieldElement) -> FieldElement:
        """Return the multiplicative inverse (reciprocal).
        
        Implements: (a/b)^(-1) = b/a
        
        Args:
            element: Field element to invert
            
        Returns:
            Multiplicative inverse
            
        Raises:
            ValueError: If element is zero
        """
        if element.is_zero():
            raise ValueError("Cannot invert zero element")
        
        ratio = element.ratio
        
        # Swap numerator and denominator
        inverse_ratio = EternalRatio(
            numerator=ratio.denominator,
            denominator=ratio.numerator
        )
        return FieldElement(ratio=inverse_ratio)
    
    def subtract(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """Subtract two field elements.
        
        Implements: a - b = a + (-b)
        
        Args:
            a: First field element
            b: Second field element
            
        Returns:
            Difference of the elements
        """
        neg_b = self.additive_inverse(b)
        return self.add(a, neg_b)
    
    def divide(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """Divide two field elements.
        
        Implements: a / b = a * b^(-1)
        
        Args:
            a: Dividend
            b: Divisor
            
        Returns:
            Quotient of the elements
            
        Raises:
            ValueError: If divisor is zero
        """
        inv_b = self.multiplicative_inverse(b)
        return self.multiply(a, inv_b)
    
    def power(self, base: FieldElement, exponent: int) -> FieldElement:
        """Raise a field element to an integer power.
        
        Args:
            base: Base element
            exponent: Integer exponent
            
        Returns:
            Power of the base
        """
        if exponent == 0:
            return self.multiplicative_identity()
        
        if exponent < 0:
            # Use multiplicative inverse for negative exponents
            inv_base = self.multiplicative_inverse(base)
            return self.power(inv_base, -exponent)
        
        # Positive exponent - repeated multiplication
        result = self.multiplicative_identity()
        current_power = base
        
        while exponent > 0:
            if exponent % 2 == 1:
                result = self.multiply(result, current_power)
            current_power = self.multiply(current_power, current_power)
            exponent //= 2
        
        return result


class EternityField:
    """Field structure for EternalRatio elements.
    
    Provides a complete field implementation with support for both
    additive and multiplicative operations, polynomial operations,
    and field-theoretic properties.
    
    Attributes:
        operation: The field operation handler
        elements: Set of field elements (for finite fields)
        characteristic: Characteristic of the field
        finite: Whether the field is finite
    
    Examples:
        >>> field = EternityField()
        >>> a = FieldElement(ratio=EternalRatio(
        ...     numerator=AbsoluteValue(2.0, 1),
        ...     denominator=AbsoluteValue(3.0, 1)
        ... ))
        >>> b = FieldElement(ratio=EternalRatio(
        ...     numerator=AbsoluteValue(1.0, 1),
        ...     denominator=AbsoluteValue(2.0, 1)
        ... ))
        >>> sum_result = field.add(a, b)
        >>> product_result = field.multiply(a, b)
    """
    
    def __init__(self, operation: Optional[EternalRatioOperation] = None,
                 elements: Optional[Set[FieldElement]] = None,
                 characteristic: int = 0,
                 finite: bool = False):
        """Initialize EternityField.
        
        Args:
            operation: Field operation handler
            elements: Set of field elements (None for infinite fields)
            characteristic: Characteristic of the field (0 for infinite characteristic)
            finite: Whether the field is finite
        """
        self.operation = operation or EternalRatioOperation()
        self.elements = elements or set()
        self.characteristic = characteristic
        self.finite = finite
        self._zero = None
        self._one = None
        self._order = None
    
    @classmethod
    def rational_field(cls, compensator: Optional[Compensator] = None) -> 'EternityField':
        """Create the field of rational EternalRatios.
        
        Args:
            compensator: Optional compensator for operations
            
        Returns:
            EternityField representing rational numbers
        """
        operation = EternalRatioOperation(compensator)
        return cls(operation=operation, characteristic=0, finite=False)
    
    @classmethod
    def finite_field(cls, prime: int, degree: int = 1,
                    compensator: Optional[Compensator] = None) -> 'EternityField':
        """Create a finite field GF(p^n).
        
        Args:
            prime: Prime characteristic
            degree: Extension degree
            compensator: Optional compensator for operations
            
        Returns:
            Finite EternityField
            
        Raises:
            ValueError: If prime is not prime or degree < 1
        """
        if not cls._is_prime(prime):
            raise ValueError(f"{prime} is not prime")
        if degree < 1:
            raise ValueError("Degree must be at least 1")
        
        operation = EternalRatioOperation(compensator)
        
        # Generate finite field elements
        elements = set()
        order = prime ** degree
        
        # For simplicity, generate elements as ratios of small integers
        for i in range(order):
            numerator = AbsoluteValue(magnitude=float(i % prime), direction=1)
            denominator = AbsoluteValue.unit_positive()
            
            ratio = EternalRatio(numerator=numerator, denominator=denominator)
            elements.add(FieldElement(ratio=ratio))
        
        field = cls(
            operation=operation,
            elements=elements,
            characteristic=prime,
            finite=True
        )
        field._order = order
        return field
    
    @staticmethod
    def _is_prime(n: int) -> bool:
        """Check if a number is prime.
        
        Args:
            n: Number to check
            
        Returns:
            True if n is prime
        """
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def add(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """Add two field elements.
        
        Args:
            a: First field element
            b: Second field element
            
        Returns:
            Sum of the elements
        """
        result = self.operation.add(a, b)
        
        if self.finite:
            # Apply modular reduction for finite fields
            result = self._reduce_modulo_characteristic(result)
            self.elements.add(result)
        
        return result
    
    def multiply(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """Multiply two field elements.
        
        Args:
            a: First field element
            b: Second field element
            
        Returns:
            Product of the elements
        """
        result = self.operation.multiply(a, b)
        
        if self.finite:
            # Apply modular reduction for finite fields
            result = self._reduce_modulo_characteristic(result)
            self.elements.add(result)
        
        return result
    
    def subtract(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """Subtract two field elements.
        
        Args:
            a: First field element
            b: Second field element
            
        Returns:
            Difference of the elements
        """
        return self.operation.subtract(a, b)
    
    def divide(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """Divide two field elements.
        
        Args:
            a: Dividend
            b: Divisor
            
        Returns:
            Quotient of the elements
        """
        return self.operation.divide(a, b)
    
    def power(self, base: FieldElement, exponent: int) -> FieldElement:
        """Raise a field element to a power.
        
        Args:
            base: Base element
            exponent: Integer exponent
            
        Returns:
            Power of the base
        """
        return self.operation.power(base, exponent)
    
    def zero(self) -> FieldElement:
        """Get the additive identity (zero).
        
        Returns:
            Zero element
        """
        if self._zero is None:
            self._zero = self.operation.additive_identity()
            if self.finite:
                self.elements.add(self._zero)
        return self._zero
    
    def one(self) -> FieldElement:
        """Get the multiplicative identity (one).
        
        Returns:
            One element
        """
        if self._one is None:
            self._one = self.operation.multiplicative_identity()
            if self.finite:
                self.elements.add(self._one)
        return self._one
    
    def additive_inverse(self, element: FieldElement) -> FieldElement:
        """Get the additive inverse of an element.
        
        Args:
            element: Element to invert
            
        Returns:
            Additive inverse
        """
        return self.operation.additive_inverse(element)
    
    def multiplicative_inverse(self, element: FieldElement) -> FieldElement:
        """Get the multiplicative inverse of an element.
        
        Args:
            element: Element to invert
            
        Returns:
            Multiplicative inverse
        """
        return self.operation.multiplicative_inverse(element)
    
    def _reduce_modulo_characteristic(self, element: FieldElement) -> FieldElement:
        """Reduce element modulo the field characteristic.
        
        Args:
            element: Element to reduce
            
        Returns:
            Reduced element
        """
        if self.characteristic == 0:
            return element
        
        # For finite fields, apply modular arithmetic
        # This is a simplified implementation
        ratio = element.ratio
        
        if not ratio.numerator.is_absolute():
            num_val = int(ratio.numerator.magnitude) % self.characteristic
            reduced_numerator = AbsoluteValue(magnitude=float(num_val), direction=ratio.numerator.direction)
        else:
            reduced_numerator = ratio.numerator
        
        if not ratio.denominator.is_absolute():
            den_val = int(ratio.denominator.magnitude) % self.characteristic
            if den_val == 0:
                den_val = 1  # Avoid zero denominator
            reduced_denominator = AbsoluteValue(magnitude=float(den_val), direction=ratio.denominator.direction)
        else:
            reduced_denominator = ratio.denominator
        
        reduced_ratio = EternalRatio(numerator=reduced_numerator, denominator=reduced_denominator)
        return FieldElement(ratio=reduced_ratio)
    
    def order(self) -> Optional[int]:
        """Get the order of the field.
        
        Returns:
            Order of the field (None if infinite)
        """
        if self.finite:
            return self._order or len(self.elements)
        return None
    
    def additive_group(self) -> AbsoluteGroup:
        """Get the additive group of the field.
        
        Returns:
            Additive group structure
        """
        # Convert field elements to group elements
        if self.finite:
            group_elements = set()
            for field_elem in self.elements:
                # Use the numerator as the group element value
                group_elem = GroupElement(value=field_elem.ratio.numerator)
                group_elements.add(group_elem)
            
            return AbsoluteGroup.additive_group().subgroup(list(group_elements))
        else:
            return AbsoluteGroup.additive_group()
    
    def multiplicative_group(self) -> AbsoluteGroup:
        """Get the multiplicative group of non-zero elements.
        
        Returns:
            Multiplicative group structure
        """
        if self.finite:
            group_elements = []
            zero = self.zero()
            
            for field_elem in self.elements:
                if field_elem != zero:
                    # Use a representative AbsoluteValue for the group
                    group_elem = GroupElement(value=field_elem.ratio.numerator)
                    group_elements.append(group_elem)
            
            return AbsoluteGroup.multiplicative_group().subgroup(group_elements)
        else:
            return AbsoluteGroup.multiplicative_group()
    
    def polynomial_ring(self, variable: str = 'x') -> 'PolynomialRing':
        """Create a polynomial ring over this field.
        
        Args:
            variable: Variable name for polynomials
            
        Returns:
            Polynomial ring over this field
        """
        return PolynomialRing(field=self, variable=variable)
    
    def is_perfect(self) -> bool:
        """Check if the field is perfect.
        
        A field is perfect if every irreducible polynomial is separable.
        
        Returns:
            True if field is perfect
        """
        # Finite fields and fields of characteristic 0 are perfect
        return self.finite or self.characteristic == 0
    
    def frobenius_endomorphism(self, element: FieldElement) -> FieldElement:
        """Apply the Frobenius endomorphism.
        
        For characteristic p > 0: φ(x) = x^p
        For characteristic 0: identity map
        
        Args:
            element: Element to apply endomorphism to
            
        Returns:
            Result of Frobenius endomorphism
        """
        if self.characteristic == 0:
            return element
        
        return self.power(element, self.characteristic)
    
    def __contains__(self, element: FieldElement) -> bool:
        """Check if element is in the field.
        
        Args:
            element: Element to check
            
        Returns:
            True if element is in field
        """
        if self.finite:
            return element in self.elements
        return True  # All EternalRatios valid for infinite field
    
    def __len__(self) -> int:
        """Return the number of elements (for finite fields).
        
        Returns:
            Number of elements
            
        Raises:
            ValueError: If field is infinite
        """
        if not self.finite:
            raise ValueError("Infinite fields have no finite length")
        return len(self.elements)
    
    def __iter__(self) -> Iterator[FieldElement]:
        """Iterate over field elements (finite fields only).
        
        Returns:
            Iterator over field elements
            
        Raises:
            ValueError: If field is infinite
        """
        if not self.finite:
            raise ValueError("Cannot iterate over infinite field")
        return iter(self.elements)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        if self.finite:
            return f"EternityField(char={self.characteristic}, order={len(self.elements)})"
        return f"EternityField(char={self.characteristic}, infinite)"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.finite:
            return f"Finite EternityField GF({self.characteristic}^{int(math.log(len(self.elements), self.characteristic))})"
        elif self.characteristic == 0:
            return "EternityField of EternalRatios (characteristic 0)"
        else:
            return f"EternityField of characteristic {self.characteristic}"


class PolynomialRing:
    """Polynomial ring over an EternityField.
    
    Provides polynomial operations over field elements,
    including addition, multiplication, division, and
    factorization capabilities.
    
    Attributes:
        field: Base field for coefficients
        variable: Variable name
    """
    
    def __init__(self, field: EternityField, variable: str = 'x'):
        """Initialize polynomial ring.
        
        Args:
            field: Base field for coefficients
            variable: Variable name for polynomials
        """
        self.field = field
        self.variable = variable
    
    def create_polynomial(self, coefficients: List[FieldElement]) -> 'Polynomial':
        """Create a polynomial from coefficients.
        
        Args:
            coefficients: List of coefficients (constant term first)
            
        Returns:
            Polynomial object
        """
        return Polynomial(coefficients=coefficients, ring=self)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"PolynomialRing({self.field}, {self.variable})"


class Polynomial:
    """Polynomial over an EternityField.
    
    Represents polynomials with coefficients from an EternityField,
    providing standard polynomial operations.
    
    Attributes:
        coefficients: List of coefficients (constant term first)
        ring: Parent polynomial ring
    """
    
    def __init__(self, coefficients: List[FieldElement], ring: PolynomialRing):
        """Initialize polynomial.
        
        Args:
            coefficients: List of coefficients (constant term first)
            ring: Parent polynomial ring
        """
        self.coefficients = self._normalize_coefficients(coefficients)
        self.ring = ring
    
    def _normalize_coefficients(self, coefficients: List[FieldElement]) -> List[FieldElement]:
        """Remove leading zero coefficients.
        
        Args:
            coefficients: Raw coefficient list
            
        Returns:
            Normalized coefficient list
        """
        if not coefficients:
            return []
        
        # Remove trailing zeros (leading coefficients)
        while len(coefficients) > 1 and coefficients[-1].is_zero():
            coefficients.pop()
        
        # If only one coefficient remains and it's zero, return empty list (zero polynomial)
        if len(coefficients) == 1 and coefficients[0].is_zero():
            return []
        
        return coefficients
    
    def degree(self) -> int:
        """Get the degree of the polynomial.
        
        Returns:
            Degree of polynomial (-1 for zero polynomial)
        """
        if not self.coefficients:
            return -1
        return len(self.coefficients) - 1
    
    def __add__(self, other: 'Polynomial') -> 'Polynomial':
        """Add two polynomials.
        
        Args:
            other: Polynomial to add
            
        Returns:
            Sum polynomial
        """
        if self.ring != other.ring:
            raise ValueError("Polynomials must be from the same ring")
        
        # Pad shorter polynomial with zeros
        max_len = max(len(self.coefficients), len(other.coefficients))
        
        self_coeffs = self.coefficients + [self.ring.field.zero()] * (max_len - len(self.coefficients))
        other_coeffs = other.coefficients + [self.ring.field.zero()] * (max_len - len(other.coefficients))
        
        # Add corresponding coefficients
        result_coeffs = []
        for i in range(max_len):
            sum_coeff = self.ring.field.add(self_coeffs[i], other_coeffs[i])
            result_coeffs.append(sum_coeff)
        
        return Polynomial(coefficients=result_coeffs, ring=self.ring)
    
    def __mul__(self, other: 'Polynomial') -> 'Polynomial':
        """Multiply two polynomials.
        
        Args:
            other: Polynomial to multiply
            
        Returns:
            Product polynomial
        """
        if self.ring != other.ring:
            raise ValueError("Polynomials must be from the same ring")
        
        if not self.coefficients or not other.coefficients:
            return Polynomial(coefficients=[], ring=self.ring)
        
        # Initialize result coefficients
        result_degree = self.degree() + other.degree()
        result_coeffs = [self.ring.field.zero()] * (result_degree + 1)
        
        # Multiply each term
        for i, a_coeff in enumerate(self.coefficients):
            for j, b_coeff in enumerate(other.coefficients):
                product = self.ring.field.multiply(a_coeff, b_coeff)
                result_coeffs[i + j] = self.ring.field.add(result_coeffs[i + j], product)
        
        return Polynomial(coefficients=result_coeffs, ring=self.ring)
    
    def evaluate(self, value: FieldElement) -> FieldElement:
        """Evaluate polynomial at a given value.
        
        Args:
            value: Value to evaluate at
            
        Returns:
            Result of evaluation
        """
        if not self.coefficients:
            return self.ring.field.zero()
        
        # Use Horner's method
        result = self.coefficients[-1]
        for i in range(len(self.coefficients) - 2, -1, -1):
            result = self.ring.field.add(
                self.ring.field.multiply(result, value),
                self.coefficients[i]
            )
        
        return result
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        if not self.coefficients:
            return "0"
        
        terms = []
        for i, coeff in enumerate(self.coefficients):
            if coeff.is_zero():
                continue
            
            if i == 0:
                terms.append(str(coeff.ratio))
            elif i == 1:
                if coeff.is_one():
                    terms.append(self.ring.variable)
                else:
                    terms.append(f"{coeff.ratio}*{self.ring.variable}")
            else:
                if coeff.is_one():
                    terms.append(f"{self.ring.variable}^{i}")
                else:
                    terms.append(f"{coeff.ratio}*{self.ring.variable}^{i}")
        
        return " + ".join(reversed(terms)) if terms else "0"