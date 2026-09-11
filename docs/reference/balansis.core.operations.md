# balansis.core.operations

[Source](../../balansis/core/operations.py) · [Reference index](index.md)

```python
class Operations:
    @staticmethod
    def compensated_add(a: AbsoluteValue, b: AbsoluteValue, compensation_factor: float=1.0) -> CompensatedResult:
        ...

    @staticmethod
    def compensated_multiply(a: AbsoluteValue, b: AbsoluteValue, compensation_factor: float=1.0) -> CompensatedResult:
        ...

    @staticmethod
    def compensated_divide(numerator: AbsoluteValue, denominator: AbsoluteValue, compensation_factor: float=1.0) -> CompensatedDivideResult:
        ...

    @staticmethod
    def compensated_divide_extended(numerator: AbsoluteValue, denominator: AbsoluteValue, compensation_factor: float=1.0) -> ExtendedDivideResult:
        ...

    @staticmethod
    def compensated_divide_policy(numerator: AbsoluteValue, denominator: AbsoluteValue, policy: SingularPolicy | str=SingularPolicy.PROPAGATE, *, compensation_factor: float=1.0, saturation_limit: float=1000000000000.0) -> PolicyDivideResult:
        ...

    @staticmethod
    def compensated_power(base: AbsoluteValue, exponent: float, compensation_factor: float=1.0) -> CompensatedResult:
        ...

    @staticmethod
    def compensated_sqrt(value: AbsoluteValue, compensation_factor: float=1.0) -> CompensatedResult:
        ...

    @staticmethod
    def compensated_log(value: AbsoluteValue, base: float=math.e, compensation_factor: float=1.0) -> CompensatedResult:
        ...

    @staticmethod
    def compensated_exp(value: AbsoluteValue, compensation_factor: float=1.0) -> CompensatedResult:
        ...

    @staticmethod
    def compensated_sin(value: AbsoluteValue, compensation_factor: float=1.0) -> CompensatedResult:
        ...

    @staticmethod
    def compensated_cos(value: AbsoluteValue, compensation_factor: float=1.0) -> CompensatedResult:
        ...

    @staticmethod
    def sequence_sum(values: List[AbsoluteValue], use_compensation: bool=True) -> CompensatedResult:
        ...

    @staticmethod
    def sequence_product(values: List[AbsoluteValue], use_compensation: bool=True) -> CompensatedResult:
        ...

    @staticmethod
    def interpolate(start: AbsoluteValue, end: AbsoluteValue, t: float) -> AbsoluteValue:
        ...

    @staticmethod
    def distance(a: AbsoluteValue, b: AbsoluteValue) -> AbsoluteValue:
        ...

    @staticmethod
    def normalize(value: AbsoluteValue) -> AbsoluteValue:
        ...
```
