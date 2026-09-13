# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.
#
# This file is part of Balansis.
# Balansis is dual-licensed under:
#   1. GNU Affero General Public License v3.0 (AGPLv3) for open-source use.
#   2. A Commercial License for proprietary and corporate use.
#
# See LICENSING.md in the project root for license selection details.
# For commercial licensing: andrew@xteam.pro
"""ACT-compensated GEMM (general matrix multiply).

``matmul`` multiplies two matrices whose entries are :class:`AbsoluteValue`,
accumulating each output entry with :meth:`Operations.compensated_multiply`
and Neumaier (improved Kahan-Babuška) summation. The result is returned
together with a scalar compensation factor that aggregates the worst-case
per-cell compensation observed during the computation.
"""

from __future__ import annotations

from typing import List, Tuple

from balansis.core.absolute import AbsoluteValue
from balansis.core.operations import Operations

Matrix = List[List[AbsoluteValue]]


def _validate_rectangular(name: str, mat: Matrix) -> int:
    if not mat:
        return 0
    width = len(mat[0])
    for row in mat:
        if len(row) != width:
            raise ValueError(f"Invalid shape for matrix {name}")
    return width


def matmul(
    a: Matrix,
    b: Matrix,
    use_compensation: bool = True,
) -> Tuple[Matrix, float]:
    """Compute C = A · B with ACT-compensated arithmetic.

    Args:
        a: Left operand, list of rows of AbsoluteValue (shape m×k).
        b: Right operand, list of rows of AbsoluteValue (shape k×n).
        use_compensation: When True (default), accumulate each row inner
            product with Neumaier compensation; otherwise use a plain
            running sum.

    Returns:
        ``(C, total_compensation)`` where ``C`` is the m×n product matrix
        (entries are :class:`AbsoluteValue`) and ``total_compensation`` is
        the maximum per-cell compensation factor observed (≥ 1.0 for
        non-empty inputs).

    Raises:
        ValueError: For inconsistent matrix shapes or mismatched inner
            dimensions.
    """
    if not a and not b:
        return [], 1.0
    if not a or not b:
        raise ValueError("Both matrices must be non-empty for matrix multiplication")

    m = len(a)
    a_cols = _validate_rectangular("a", a)
    if len(b) != a_cols:
        raise ValueError("Inner dimensions must match")
    n = _validate_rectangular("b", b)

    result: Matrix = []
    overall_compensation = 1.0

    for i in range(m):
        row_res: List[AbsoluteValue] = []
        for j in range(n):
            total = 0.0
            compensation_sum = 0.0
            cell_compensation = 1.0
            for k in range(a_cols):
                prod, prod_comp = Operations.compensated_multiply(a[i][k], b[k][j])
                if prod_comp > cell_compensation:
                    cell_compensation = prod_comp
                v = prod.to_float()
                if use_compensation:
                    t = total + v
                    if abs(total) >= abs(v):
                        compensation_sum += (total - t) + v
                    else:
                        compensation_sum += (v - t) + total
                    total = t
                else:
                    total = total + v
            corrected = total + compensation_sum if use_compensation else total
            row_res.append(AbsoluteValue.from_float(corrected))
            if cell_compensation > overall_compensation:
                overall_compensation = cell_compensation
        result.append(row_res)

    return result, overall_compensation
