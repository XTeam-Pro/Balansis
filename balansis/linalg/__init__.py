# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.
#
# This file is part of Balansis.
# Balansis is dual-licensed under:
#   1. GNU Affero General Public License v3.0 (AGPLv3) for open-source use.
#   2. A Commercial License for proprietary and corporate use.
#
# See LICENSING.md in the project root for license selection details.
# For commercial licensing: andrew@xteam.pro
"""ACT-compensated linear algebra primitives.

Public API:
    matmul           — ACT-compensated GEMM (returns ``(C, compensation)``)
    qr_decompose     — QR decomposition via Householder/Givens/MGS
    svd              — Singular Value Decomposition
    CompensatedQRResult — structured QR result with diagnostics
    CompensatedSVDResult — structured SVD result with diagnostics
"""

from .gemm import matmul
from .qr import CompensatedQRResult, qr_decompose
from .svd import CompensatedSVDResult, svd

__all__ = [
    "matmul",
    "qr_decompose",
    "svd",
    "CompensatedQRResult",
    "CompensatedSVDResult",
]
