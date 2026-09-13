# Matrix operations

Matrices are rectangular nested lists of `AbsoluteValue`. `matmul(a, b)` returns
`(matrix, compensation_factor)`. Products use `Operations.compensated_multiply`;
`use_compensation` selects compensated or ordinary accumulation of those products.

`qr_decompose(a, method="householder")` returns a `CompensatedQRResult` with `Q`,
`R`, `orthogonality_error`, `method` and `compensation_factors`. Methods are
`householder`, `givens` and `gram_schmidt`. The result can be unpacked as `Q, R`.

`svd(a, method="numpy_gesdd")` returns `U`, `S`, `Vt`, `reconstruction_error`,
`compensation_factors` and `singular_events`. It can be unpacked as `U, S, Vt`.
`act_jacobi` uses one-sided Jacobi rotations, exact Gram accumulation and
contiguous column storage. Very large and small matrices are scaled for the
iteration; zero singular vectors receive an orthonormal basis completion.

```python
import numpy as np
from balansis import B
from balansis.linalg import svd, qr_decompose

a = [[B(1), B(2)], [B(3), B(4)], [B(5), B(6)]]
result = svd(a, method="act_jacobi")
u = np.array([[v.to_float() for v in row] for row in result.U])
s = np.array([v.to_float() for v in result.S])
vt = np.array([[v.to_float() for v in row] for row in result.Vt])
assert np.allclose(u @ np.diag(s) @ vt, [[1, 2], [3, 4], [5, 6]])
assert qr_decompose(a).orthogonality_error < 1e-12
```

SVD singular policies apply to the ratio of the leading singular value to each
singular value. `singular_telemetry()` serializes the resulting events.
[Source](../balansis/linalg).
