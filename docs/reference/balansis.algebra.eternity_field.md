# balansis.algebra.eternity_field

[Source](../../balansis/algebra/eternity_field.py) · [Reference index](index.md)

```python
class FieldElement(BaseModel):
    ratio: EternalRatio

    additive_order: Optional[int] = Field(default=None, description='Order in additive group')

    multiplicative_order: Optional[int] = Field(default=None, description='Order in multiplicative group')

    minimal_polynomial: Optional[List[float]] = Field(default=None, description='Minimal polynomial coefficients')

    @classmethod
    def validate_orders(cls, v: Optional[int]) -> Optional[int]:
        ...

    def is_zero(self) -> bool:
        ...

    def is_one(self) -> bool:
        ...

    def is_unit(self) -> bool:
        ...
```

```python
class FieldOperation(ABC):
    def add(self, a: FieldElement, b: FieldElement) -> FieldElement:
        ...

    def multiply(self, a: FieldElement, b: FieldElement) -> FieldElement:
        ...

    def additive_identity(self) -> FieldElement:
        ...

    def multiplicative_identity(self) -> FieldElement:
        ...

    def additive_inverse(self, element: FieldElement) -> FieldElement:
        ...

    def multiplicative_inverse(self, element: FieldElement) -> FieldElement:
        ...
```

```python
class EternalRatioOperation(FieldOperation):
    def __init__(self, compensator: Optional[Compensator]=None):
        ...

    def add(self, a: FieldElement, b: FieldElement) -> FieldElement:
        ...

    def multiply(self, a: FieldElement, b: FieldElement) -> FieldElement:
        ...

    def additive_identity(self) -> FieldElement:
        ...

    def multiplicative_identity(self) -> FieldElement:
        ...

    def additive_inverse(self, element: FieldElement) -> FieldElement:
        ...

    def multiplicative_inverse(self, element: FieldElement) -> FieldElement:
        ...

    def subtract(self, a: FieldElement, b: FieldElement) -> FieldElement:
        ...

    def divide(self, a: FieldElement, b: FieldElement) -> FieldElement:
        ...

    def power(self, base: FieldElement, exponent: int) -> FieldElement:
        ...
```

```python
class PrimeFieldOperation(EternalRatioOperation):
    def __init__(self, prime: int):
        ...

    def residue(self, element: FieldElement) -> int:
        ...

    def element(self, value: int) -> FieldElement:
        ...

    def add(self, a: FieldElement, b: FieldElement) -> FieldElement:
        ...

    def multiply(self, a: FieldElement, b: FieldElement) -> FieldElement:
        ...

    def additive_inverse(self, element: FieldElement) -> FieldElement:
        ...

    def multiplicative_inverse(self, element: FieldElement) -> FieldElement:
        ...
```

```python
class FieldGroupOperation(GroupOperation):
    def __init__(self, field: EternityField, multiplicative: bool):
        ...

    def apply(self, a: GroupElement, b: GroupElement) -> GroupElement:
        ...

    def identity(self) -> GroupElement:
        ...

    def inverse(self, element: GroupElement) -> GroupElement:
        ...
```

```python
class EternityField:
    def __init__(self, operation: Optional[EternalRatioOperation]=None, elements: Optional[Set[FieldElement]]=None, characteristic: int=0, finite: bool=False):
        ...

    @classmethod
    def rational_field(cls, compensator: Optional[Compensator]=None) -> 'EternityField':
        ...

    @classmethod
    def finite_field(cls, prime: int, degree: int=1, compensator: Optional[Compensator]=None) -> 'EternityField':
        ...

    def add(self, a: FieldElement, b: FieldElement) -> FieldElement:
        ...

    def multiply(self, a: FieldElement, b: FieldElement) -> FieldElement:
        ...

    def subtract(self, a: FieldElement, b: FieldElement) -> FieldElement:
        ...

    def divide(self, a: FieldElement, b: FieldElement) -> FieldElement:
        ...

    def power(self, base: FieldElement, exponent: int) -> FieldElement:
        ...

    def zero(self) -> FieldElement:
        ...

    def one(self) -> FieldElement:
        ...

    def additive_inverse(self, element: FieldElement) -> FieldElement:
        ...

    def multiplicative_inverse(self, element: FieldElement) -> FieldElement:
        ...

    def order(self) -> Optional[int]:
        ...

    def additive_group(self) -> AbsoluteGroup:
        ...

    def multiplicative_group(self) -> AbsoluteGroup:
        ...

    def polynomial_ring(self, variable: str='x') -> 'PolynomialRing':
        ...

    def is_perfect(self) -> bool:
        ...

    def frobenius_endomorphism(self, element: FieldElement) -> FieldElement:
        ...
```

```python
class PolynomialRing:
    def __init__(self, field: EternityField, variable: str='x'):
        ...

    def create_polynomial(self, coefficients: List[FieldElement]) -> 'Polynomial':
        ...
```

```python
class Polynomial:
    def __init__(self, coefficients: List[FieldElement], ring: PolynomialRing):
        ...

    def degree(self) -> int:
        ...

    def evaluate(self, value: FieldElement) -> FieldElement:
        ...
```
