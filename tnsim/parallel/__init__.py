"""Backward-compatible parallel namespace for TNSIM."""

from ..core.operations.parallel_tnsim import (
    ParallelTNSIM,
    get_global_parallel_processor,
)

__all__ = ["ParallelTNSIM", "get_global_parallel_processor"]
