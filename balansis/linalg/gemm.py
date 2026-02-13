"""General matrix multiplication with ACT compensation.

This module implements matrix multiplication using compensated arithmetic
operations to maintain numerical stability according to ACT principles.
"""

from typing import List, Tuple

from balansis.core.absolute import AbsoluteValue
from balansis.core.operations import Operations

# Type alias for matrix of AbsoluteValues
Matrix = List[List[AbsoluteValue]]
# Type alias for compensated matrix result
CompensatedMatrixResult = Tuple[Matrix, float]


def matmul(
    a: Matrix,
    b: Matrix,
    use_compensation: bool = True,
) -> CompensatedMatrixResult:
    """Perform general matrix multiplication using ACT-compensated operations.

    Computes C = A * B using compensated multiplication and addition
    to maintain numerical stability throughout the computation.

    Args:
        a: Left matrix of AbsoluteValues with shape (n, m).
        b: Right matrix of AbsoluteValues with shape (m, p).
        use_compensation: Whether to use ACT-compensated operations.
            When False, uses standard arithmetic for performance.

    Returns:
        Tuple of (result_matrix, total_compensation_factor) where
        result_matrix has shape (n, p) and total_compensation_factor
        is the accumulated compensation across all element computations.

    Raises:
        ValueError: If matrices are empty or have incompatible shapes.
    """
    if not a or not b:
        return [], 1.0
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

    result: Matrix = []
    total_compensation = 1.0

    for i in range(n):
        row_res: List[AbsoluteValue] = []
        for j in range(p):
            acc = AbsoluteValue.absolute()
            element_compensation = 1.0

            for k in range(m):
                if use_compensation:
                    prod, mul_comp = Operations.compensated_multiply(
                        a[i][k], b[k][j]
                    )
                    acc, add_comp = Operations.compensated_add(acc, prod)
                    element_compensation *= mul_comp * add_comp
                else:
                    prod = a[i][k] * b[k][j].to_float()
                    acc = acc + prod

            row_res.append(acc)
            total_compensation *= element_compensation
        result.append(row_res)

    return result, total_compensation
