# balansis.core.eternity

[Source](../../balansis/core/eternity.py) · [Reference index](index.md)

```python
class EternalRatio(BaseModel):
    numerator: AbsoluteValue = Field(..., description='AbsoluteValue in the numerator position')

    denominator: AbsoluteValue = Field(..., description='AbsoluteValue in the denominator position (non-Absolute)')

    @classmethod
    def denominator_not_absolute(cls, v: AbsoluteValue) -> AbsoluteValue:
        ...

    def value(self) -> float:
        ...

    def signed_value(self) -> float:
        ...

    def is_stable(self, tolerance: float=1e-10) -> bool:
        ...

    def to_json(self) -> dict[str, Any]:
        ...

    def inverse(self) -> 'EternalRatio':
        ...

    def reciprocal(self) -> 'EternalRatio':
        ...

    def log(self) -> float:
        ...

    def exp(self) -> 'EternalRatio':
        ...

    def sin(self) -> float:
        ...

    def cos(self) -> float:
        ...

    def tan(self) -> float:
        ...

    def power(self, exponent: float) -> 'EternalRatio':
        ...

    def simplify(self, tolerance: float=1e-10) -> 'EternalRatio':
        ...

    def to_absolute_value(self) -> AbsoluteValue:
        ...

    @classmethod
    def from_float(cls, value: float) -> 'EternalRatio':
        ...

    @classmethod
    def from_values(cls, numerator_value: float, denominator_value: float) -> 'EternalRatio':
        ...

    @classmethod
    def unity(cls) -> 'EternalRatio':
        ...

    def is_unity(self, tolerance: float=1e-10) -> bool:
        ...

    def is_integer(self, tolerance: float=1e-10) -> bool:
        ...

    def numerical_value(self) -> float:
        ...

    def is_reciprocal(self, other: 'EternalRatio', tolerance: float=1e-10) -> bool:
        ...
```

```python
class SingularPolicy(str, Enum):
    ...
```

```python
class SingularArithmeticEvent(BaseModel):
    operation: str = Field(..., description='Operation that produced or handled the singular state')

    policy: SingularPolicy = Field(..., description='Applied singular arithmetic policy')

    input_kind: str = Field(..., description='Original ExtendedRatio kind before policy application')

    output_kind: str = Field(..., description='Resulting kind after policy application')

    reason: Optional[str] = Field(default=None, description='Optional explanation of the event')

    direction: Optional[int] = Field(default=None, description='Direction for infinite states')

    saturated: bool = Field(default=False, description='Whether the event saturated an infinite state')

    numeric_value: Optional[float] = Field(default=None, description='Numeric output value when finite or infinite')
```

```python
class ExtendedRatio(BaseModel):
    kind: Literal['finite', 'infinite', 'indeterminate'] = Field(..., description='Semantic state of the ratio')

    ratio: Optional[EternalRatio] = Field(default=None, description="Finite ratio payload when kind='finite'")

    direction: Optional[int] = Field(default=None, description="Direction of infinity when kind='infinite'")

    reason: Optional[str] = Field(default=None, description='Optional explanation for singular or indeterminate states')

    @classmethod
    def direction_must_be_valid(cls, value: Optional[int]) -> Optional[int]:
        ...

    def validate_state(self) -> 'ExtendedRatio':
        ...

    @classmethod
    def from_ratio(cls, ratio: EternalRatio) -> 'ExtendedRatio':
        ...

    @classmethod
    def from_float(cls, value: float) -> 'ExtendedRatio':
        ...

    @classmethod
    def positive_infinity(cls, reason: Optional[str]=None) -> 'ExtendedRatio':
        ...

    @classmethod
    def negative_infinity(cls, reason: Optional[str]=None) -> 'ExtendedRatio':
        ...

    @classmethod
    def indeterminate(cls, reason: Optional[str]=None) -> 'ExtendedRatio':
        ...

    @classmethod
    def from_division(cls, numerator: AbsoluteValue, denominator: AbsoluteValue, reason: Optional[str]=None) -> 'ExtendedRatio':
        ...

    def is_finite(self) -> bool:
        ...

    def is_infinite(self) -> bool:
        ...

    def is_indeterminate(self) -> bool:
        ...

    def is_singular(self) -> bool:
        ...

    def is_stable(self, tolerance: float=1e-10) -> bool:
        ...

    def numerical_value(self) -> float:
        ...

    def signed_value(self) -> float:
        ...

    def finite_ratio(self) -> EternalRatio:
        ...

    def saturate(self, limit: float=1000000000000.0) -> 'ExtendedRatio':
        ...

    def policy_event(self, operation: str, policy: SingularPolicy | str, result: Optional['ExtendedRatio']=None) -> SingularArithmeticEvent:
        ...

    def apply_policy(self, policy: SingularPolicy | str=SingularPolicy.PROPAGATE, *, operation: str='extended_ratio', saturation_limit: float=1000000000000.0) -> tuple['ExtendedRatio', Optional[SingularArithmeticEvent]]:
        ...

    def inverse(self) -> 'ExtendedRatio':
        ...

    def to_json(self) -> dict[str, Any]:
        ...
```
