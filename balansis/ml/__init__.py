"""Machine learning integration module for Balansis.

Provides ACT-aware optimizers for training neural networks with
enhanced numerical stability.
"""

from .optimizer import EternalOptimizer

__all__ = ["EternalOptimizer"]

try:
    import torch  # noqa: F401
    from .optimizer import EternalTorchOptimizer

    __all__.append("EternalTorchOptimizer")
except ImportError:
    pass
