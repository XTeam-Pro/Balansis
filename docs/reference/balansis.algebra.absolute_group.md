# balansis.algebra.absolute_group

[Source](../../balansis/algebra/absolute_group.py) · [Reference index](index.md)

```python
class GroupElement(BaseModel):
    value: AbsoluteValue

    order: Optional[int] = Field(default=None, description='Order of element in group')

    conjugacy_class: Optional[str] = Field(default=None, description='Conjugacy class ID')

    @classmethod
    def validate_order(cls, v: Optional[int]) -> Optional[int]:
        ...
```

```python
class GroupOperation(ABC):
    def apply(self, a: GroupElement, b: GroupElement) -> GroupElement:
        ...

    def identity(self) -> GroupElement:
        ...

    def inverse(self, element: GroupElement) -> GroupElement:
        ...
```

```python
class AdditiveOperation(GroupOperation):
    def __init__(self, compensator: Optional[Compensator]=None):
        ...

    def apply(self, a: GroupElement, b: GroupElement) -> GroupElement:
        ...

    def identity(self) -> GroupElement:
        ...

    def inverse(self, element: GroupElement) -> GroupElement:
        ...

    def operate(self, a: GroupElement, b: GroupElement) -> GroupElement:
        ...

    def identity_element(self) -> GroupElement:
        ...

    def inverse_element(self, element: GroupElement) -> GroupElement:
        ...
```

```python
class MultiplicativeOperation(GroupOperation):
    def __init__(self, compensator: Optional[Compensator]=None):
        ...

    def apply(self, a: GroupElement, b: GroupElement) -> GroupElement:
        ...

    def identity(self) -> GroupElement:
        ...

    def inverse(self, element: GroupElement) -> GroupElement:
        ...

    def operate(self, a: GroupElement, b: GroupElement) -> GroupElement:
        ...

    def identity_element(self) -> GroupElement:
        ...

    def inverse_element(self, element: GroupElement) -> GroupElement:
        ...
```

```python
class CyclicOperation(AdditiveOperation):
    def __init__(self, modulus: int):
        ...

    def apply(self, a: GroupElement, b: GroupElement) -> GroupElement:
        ...

    def inverse(self, element: GroupElement) -> GroupElement:
        ...
```

```python
class AbsoluteGroup:
    def __init__(self, operation: GroupOperation, elements: Optional[Set[GroupElement]]=None, finite: bool=False):
        ...

    @classmethod
    def additive_group(cls, compensator: Optional[Compensator]=None) -> 'AbsoluteGroup':
        ...

    @classmethod
    def multiplicative_group(cls, compensator: Optional[Compensator]=None) -> 'AbsoluteGroup':
        ...

    @classmethod
    def finite_cyclic_group(cls, order: int, compensator: Optional[Compensator]=None) -> 'AbsoluteGroup':
        ...

    def operate(self, a: GroupElement, b: GroupElement) -> GroupElement:
        ...

    def identity_element(self) -> GroupElement:
        ...

    def inverse_element(self, element: GroupElement) -> GroupElement:
        ...

    def order(self) -> Optional[int]:
        ...

    def element_order(self, element: GroupElement) -> Optional[int]:
        ...

    def is_abelian(self) -> bool:
        ...

    def subgroup(self, generators: List[GroupElement]) -> 'AbsoluteGroup':
        ...

    def cosets(self, subgroup: 'AbsoluteGroup', left: bool=True) -> List[Set[GroupElement]]:
        ...

    def is_normal_subgroup(self, subgroup: 'AbsoluteGroup') -> bool:
        ...

    def quotient_group(self, normal_subgroup: 'AbsoluteGroup') -> 'AbsoluteGroup':
        ...
```
