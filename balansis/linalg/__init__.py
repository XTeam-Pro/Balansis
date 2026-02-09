"""Linear algebra module for Balansis.

Provides ACT-compensated linear algebra operations including matrix
multiplication, singular value decomposition, and QR decomposition.
"""

from .gemm import matmul
from .svd import svd
from .qr import qr_decompose

__all__ = ["matmul", "svd", "qr_decompose"]
