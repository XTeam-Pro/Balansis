"""Core modules for TNSIM (Theory of Null-Sum Infinite Multitudes)."""

from .cache.tnsim_cache import TNSIMCache, cached_operation, get_global_cache
from .operations.parallel_tnsim import ParallelTNSIM, get_global_parallel_processor
from .sets.zero_sum_infinite_set import ZeroSumInfiniteSet

__all__ = [
    "ZeroSumInfiniteSet",
    "TNSIMCache",
    "cached_operation",
    "get_global_cache",
    "ParallelTNSIM",
    "get_global_parallel_processor",
]

from .. import __version__

__author__ = "Andrey Tikhonov"
__description__ = "Theory of Null-Sum Infinite Multitudes - Core Modules"
