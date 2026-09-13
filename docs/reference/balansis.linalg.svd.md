# balansis.linalg.svd

[Source](../../balansis/linalg/svd.py) · [Reference index](index.md)

```python
class CompensatedSVDResult:
    U: Matrix

    S: Vector

    Vt: Matrix

    reconstruction_error: float

    method: str

    compensation_factors: List[float] = field(default_factory=list)

    singular_events: List[SingularArithmeticEvent] = field(default_factory=list)

    def singular_telemetry(self) -> List[dict[str, object]]:
        ...
```

```python
def svd(a: Matrix, method: str='numpy_gesdd', singular_policy: SingularPolicy | str=SingularPolicy.PROPAGATE, saturation_limit: float=1000000000000.0) -> CompensatedSVDResult:
    ...
```
