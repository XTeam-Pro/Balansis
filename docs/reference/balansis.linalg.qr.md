# balansis.linalg.qr

[Source](../../balansis/linalg/qr.py) · [Reference index](index.md)

```python
class CompensatedQRResult:
    Q: Matrix

    R: Matrix

    orthogonality_error: float

    method: str

    compensation_factors: List[float] = field(default_factory=list)
```

```python
def qr_decompose(a: Matrix, method: str='householder') -> CompensatedQRResult:
    ...
```
