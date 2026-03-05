# Copyright (c) 2024-2026 Tikhonov Andrey. All rights reserved.
# SPDX-License-Identifier: MIT (non-commercial) | Commercial use: see COMMERCIAL_LICENSE.md
from typing import List, Tuple
from balansis.core.absolute import AbsoluteValue

def svd(a: List[List[AbsoluteValue]]):
    try:
        import numpy as np
    except ImportError:
        raise ImportError("numpy required for svd")
    A = np.array([[x.to_float() for x in row] for row in a], dtype=float)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    U_abs = [[AbsoluteValue.from_float(U[i][j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
    S_abs = [AbsoluteValue.from_float(s) for s in S]
    Vt_abs = [[AbsoluteValue.from_float(Vt[i][j]) for j in range(Vt.shape[1])] for i in range(Vt.shape[0])]
    return U_abs, S_abs, Vt_abs
