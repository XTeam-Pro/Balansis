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
