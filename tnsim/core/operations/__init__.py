"""Operations module for TNSIM parallel computations."""

from .parallel_tnsim import ParallelTNSIM, get_global_parallel_processor

__all__ = ["ParallelTNSIM", "get_global_parallel_processor"]
