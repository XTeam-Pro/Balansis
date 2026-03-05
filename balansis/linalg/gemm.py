# Copyright (c) 2024-2026 Tikhonov Andrey. All rights reserved.
# SPDX-License-Identifier: MIT (non-commercial) | Commercial use: see COMMERCIAL_LICENSE.md
from typing import List
from balansis.core.absolute import AbsoluteValue

def matmul(a: List[List[AbsoluteValue]], b: List[List[AbsoluteValue]]) -> List[List[AbsoluteValue]]:
    if not a or not b:
        return []
    n = len(a)
    m = len(a[0])
    p = len(b[0])
    for row in a:
        if len(row) != m:
            raise ValueError("Invalid shape for matrix a")
    if len(b) != m:
        raise ValueError("Inner dimensions must match")
    for row in b:
        if len(row) != p:
            raise ValueError("Invalid shape for matrix b")
    result: List[List[AbsoluteValue]] = []
    for i in range(n):
        row_res: List[AbsoluteValue] = []
        for j in range(p):
            acc = AbsoluteValue.absolute()
            for k in range(m):
                prod = a[i][k] * b[k][j].to_float()
                acc = acc + prod
            row_res.append(acc)
        result.append(row_res)
    return result
