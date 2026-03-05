# Copyright (c) 2024-2026 Tikhonov Andrey. All rights reserved.
# SPDX-License-Identifier: MIT (non-commercial) | Commercial use: see COMMERCIAL_LICENSE.md
from typing import List, Tuple
from balansis.core.absolute import AbsoluteValue

def qr_decompose(a: List[List[AbsoluteValue]]) -> Tuple[List[List[AbsoluteValue]], List[List[AbsoluteValue]]]:
    m = len(a)
    n = len(a[0]) if m else 0
    A = [[x.to_float() for x in row] for row in a]
    Q = [[0.0 for _ in range(n)] for _ in range(m)]
    R = [[0.0 for _ in range(n)] for _ in range(n)]
    for j in range(n):
        v = [A[i][j] for i in range(m)]
        for k in range(j):
            R[k][j] = sum(Q[i][k]*A[i][j] for i in range(m))
            for i in range(m):
                v[i] -= R[k][j]*Q[i][k]
        norm = sum(v[i]*v[i] for i in range(m))**0.5
        if norm == 0.0:
            for i in range(m):
                Q[i][j] = 0.0
            R[j][j] = 0.0
        else:
            for i in range(m):
                Q[i][j] = v[i]/norm
            R[j][j] = norm
    Q_abs = [[AbsoluteValue.from_float(Q[i][j]) for j in range(n)] for i in range(m)]
    R_abs = [[AbsoluteValue.from_float(R[i][j]) for j in range(n)] for i in range(n)]
    return Q_abs, R_abs
