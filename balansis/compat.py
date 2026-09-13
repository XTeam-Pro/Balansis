# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.
#
# This file is part of Balansis.
# Balansis is dual-licensed under:
#   1. GNU Affero General Public License v3.0 (AGPLv3) for open-source use.
#   2. A Commercial License for proprietary and corporate use.
#
# See LICENSING.md in the project root for license selection details.
# For commercial licensing: andrew@xteam.pro
from __future__ import annotations

from typing import Any

import numpy as np

from balansis.linalg.gemm import matmul
from balansis.numpy_integration import compensated_softmax


def _torch() -> Any:
    import sys

    return sys.modules.get("torch")


class CompensatedSum:
    """Compensated scalar reduction for NumPy arrays and Torch tensors."""

    def __init__(self, tolerance: float = 1e-12) -> None:
        self.tolerance = tolerance

    def __call__(self, values: Any) -> Any:
        torch = _torch()
        if torch is not None and isinstance(values, torch.Tensor):
            if not torch.isfinite(values).all():
                raise ValueError("CompensatedSum requires finite values")
            flat = values.to(dtype=torch.float64).reshape(-1)
            total = flat.new_zeros(())
            correction = flat.new_zeros(())
            for value in flat:
                updated = total + value
                correction = correction + torch.where(
                    total.abs() >= value.abs(),
                    (total - updated) + value,
                    (value - updated) + total,
                )
                total = updated
            return total + correction
        from balansis.array import sum_array

        result, _ = sum_array(np.asarray(values).reshape(-1))
        return result.to_float()


class StableSoftmax:
    """Compatibility wrapper providing numerically stable softmax."""

    def __call__(self, logits: Any) -> Any:
        torch = _torch()
        if torch is not None and isinstance(logits, torch.Tensor):
            return torch.softmax(logits, dim=-1)
        return np.apply_along_axis(
            compensated_softmax, -1, np.asarray(logits, dtype=np.float64)
        )


class CompensatedMatMul:
    """Compatibility wrapper for matrix multiplication across tensor types."""

    def __call__(self, left: Any, right: Any) -> Any:
        torch = _torch()
        if torch is not None and isinstance(left, torch.Tensor):
            return torch.matmul(left, right)

        left_arr = np.asarray(left)
        right_arr = np.asarray(right)
        if left_arr.dtype.kind in "biuf" and right_arr.dtype.kind in "biuf":
            return np.matmul(left_arr, right_arr)

        # Fall back to the ACT matrix multiply for AbsoluteValue inputs.
        product, _ = matmul(left, right)
        return product
