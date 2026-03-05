"""Machine learning integration module for Balansis.

Provides ACT-aware optimizers for training neural networks with
enhanced numerical stability.
"""

from .optimizer import EternalOptimizer, AdaptiveEternalOptimizer

__all__ = ["EternalOptimizer", "AdaptiveEternalOptimizer"]

try:
    import torch  # noqa: F401
    from .optimizer import EternalTorchOptimizer

    __all__.append("EternalTorchOptimizer")
except ImportError:
    pass
