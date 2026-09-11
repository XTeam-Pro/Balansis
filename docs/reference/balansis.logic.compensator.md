# balansis.logic.compensator

[Source](../../balansis/logic/compensator.py) · [Reference index](index.md)

```python
class CompensationType(Enum):
    ...
```

```python
class CompensationRecord:
    operation_type: str

    compensation_type: CompensationType

    original_values: List[Any]

    compensated_values: List[Any]

    compensation_factor: float

    stability_metric: float

    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        ...
```

```python
class CompensationStrategy(BaseModel):
    stability_threshold: float = Field(default=1e-12, description='Threshold below which stability compensation is applied')

    overflow_threshold: float = Field(default=1e+100, description='Threshold above which overflow compensation is applied')

    underflow_threshold: float = Field(default=1e-100, description='Threshold below which underflow compensation is applied')

    max_iterations: int = Field(default=100, description='Maximum iterations for convergence compensation')

    convergence_tolerance: float = Field(default=1e-10, description='Tolerance for convergence checks')

    balance_factor: float = Field(default=0.5, description='Factor for balance compensation (0.0 to 1.0)')

    max_history_size: int = Field(default=10000, gt=0, description='Maximum number of compensation records to keep in history')

    @classmethod
    def validate_balance_factor(cls, v: float) -> float:
        ...

    @classmethod
    def high_precision(cls) -> 'CompensationStrategy':
        ...

    @classmethod
    def balanced(cls) -> 'CompensationStrategy':
        ...

    @classmethod
    def fast(cls) -> 'CompensationStrategy':
        ...
```

```python
class Compensator:
    def __init__(self, strategy: Optional[CompensationStrategy]=None):
        ...

    def analyze_stability(self, values: List[AbsoluteValue]) -> float:
        ...

    def detect_compensation_need(self, operation: str, operands: List[AbsoluteValue]) -> List[CompensationType]:
        ...

    def apply_stability_compensation(self, values: List[AbsoluteValue]) -> List[AbsoluteValue]:
        ...

    def apply_overflow_compensation(self, values: List[AbsoluteValue]) -> List[AbsoluteValue]:
        ...

    def apply_balance_compensation(self, a: AbsoluteValue, b: AbsoluteValue) -> Tuple[AbsoluteValue, AbsoluteValue]:
        ...

    def compensate_addition(self, a: AbsoluteValue, b: AbsoluteValue) -> AbsoluteValue:
        ...

    def compensate_multiplication(self, a: AbsoluteValue, b: AbsoluteValue) -> AbsoluteValue:
        ...

    def compensate_division(self, numerator: AbsoluteValue, denominator: AbsoluteValue) -> EternalRatio:
        ...

    def compensate_division_extended(self, numerator: AbsoluteValue, denominator: AbsoluteValue) -> ExtendedRatio:
        ...

    def compensate_division_policy(self, numerator: AbsoluteValue, denominator: AbsoluteValue, policy: SingularPolicy | str=SingularPolicy.PROPAGATE, *, saturation_limit: float=1000000000000.0) -> tuple[ExtendedRatio, Optional[SingularArithmeticEvent]]:
        ...

    def compensate_power(self, base: AbsoluteValue, exponent: float) -> AbsoluteValue:
        ...

    def compensate_sequence(self, operation: Callable, values: List[AbsoluteValue]) -> AbsoluteValue:
        ...

    def get_compensation_summary(self) -> Dict[str, Any]:
        ...

    def get_singular_telemetry(self) -> Dict[str, Any]:
        ...

    def reset_history(self) -> None:
        ...

    def set_strategy(self, strategy: CompensationStrategy) -> None:
        ...

    def compensate_array(self, arr_a: List[AbsoluteValue], arr_b: List[AbsoluteValue]) -> List[AbsoluteValue]:
        ...
```
