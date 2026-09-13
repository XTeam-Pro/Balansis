# balansis.core.absolute

[Source](../../balansis/core/absolute.py) · [Reference index](index.md)

```python
class AbsoluteValue(BaseModel):
    magnitude: Magnitude = Field(..., ge=0.0, description='Non-negative magnitude of the absolute value')

    direction: Direction = Field(..., description='Direction indicator: +1 for positive, -1 for negative')

    @classmethod
    def magnitude_must_be_finite(cls, v: float) -> float:
        ...

    @classmethod
    def direction_must_be_valid(cls, v: int) -> int:
        ...

    def to_json(self) -> dict[str, Any]:
        ...

    def inverse(self) -> 'AbsoluteValue':
        ...

    def log(self) -> float:
        ...

    def exp(self) -> 'AbsoluteValue':
        ...

    def sin(self) -> float:
        ...

    def cos(self) -> float:
        ...

    def tan(self) -> float:
        ...

    def to_float(self) -> float:
        ...

    def is_absolute(self) -> bool:
        ...

    def is_positive(self) -> bool:
        ...

    def is_negative(self) -> bool:
        ...

    def is_unit(self) -> bool:
        ...

    def is_compensating(self) -> bool:
        ...

    def compensates_with(self, other: 'AbsoluteValue') -> bool:
        ...

    @classmethod
    def from_float(cls, value: float) -> 'AbsoluteValue':
        ...

    @classmethod
    def absolute(cls) -> 'AbsoluteValue':
        ...

    @classmethod
    def unit_positive(cls) -> 'AbsoluteValue':
        ...

    @classmethod
    def unit_negative(cls) -> 'AbsoluteValue':
        ...
```
