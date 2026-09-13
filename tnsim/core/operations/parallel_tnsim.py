"""Parallel computations for TNSIM operations."""

import asyncio
import concurrent.futures
import threading
from datetime import datetime
from decimal import Decimal
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional

from ..cache.tnsim_cache import TNSIMCache

# Import local modules
from ..sets.zero_sum_infinite_set import ZeroSumInfiniteSet


class ParallelTNSIM:
    """Class for parallel execution of TNSIM operations.

    Provides efficient execution of ⊕ operations on multiple sets
    of infinite sets using multithreading and asynchronous processing.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_cache: bool = True,
        chunk_size: int = 100,
    ):
        """Initialize parallel processor.

        Args:
            max_workers: Maximum number of worker threads
            use_cache: Whether to use caching
            chunk_size: Chunk size for processing
        """
        if max_workers is not None and max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.max_workers = max_workers or min(32, (cpu_count() or 1) + 4)
        self.use_cache = use_cache
        self.chunk_size = chunk_size

        # Initialize cache
        self.cache = TNSIMCache() if use_cache else None

        # Execution pools
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        )
        self.process_pool = None  # Initialized on demand

        # Execution statistics
        self.stats = {
            "operations_completed": 0,
            "total_execution_time": 0,
            "parallel_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        self._lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def shutdown(self):
        """Shutdown pools."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.close()
            self.process_pool.join()

    async def parallel_zero_sum_operations(
        self,
        set_pairs: List[tuple[ZeroSumInfiniteSet, ZeroSumInfiniteSet]],
        method: str = "compensated",
    ) -> List[Decimal]:
        """Parallel execution of ⊕ operations for multiple set pairs.

        Args:
            set_pairs: List of set pairs for operations
            method: Calculation method

        Returns:
            List of ⊕ operation results
        """
        start_time = datetime.now()

        # Split into chunks
        chunks = self._create_chunks(set_pairs, self.chunk_size)

        # Create tasks for asynchronous execution
        tasks = []
        for chunk in chunks:
            task = asyncio.create_task(self._process_chunk_async(chunk, method))
            tasks.append(task)

        # Wait for all tasks to complete
        chunk_results = await asyncio.gather(*tasks)

        # Combine results
        results = []
        for chunk_result in chunk_results:
            results.extend(chunk_result)

        # Update statistics
        execution_time = (datetime.now() - start_time).total_seconds()
        with self._lock:
            self.stats["operations_completed"] += len(set_pairs)
            self.stats["total_execution_time"] += execution_time
            self.stats["parallel_operations"] += 1

        return results

    async def _process_chunk_async(
        self, chunk: List[tuple[ZeroSumInfiniteSet, ZeroSumInfiniteSet]], method: str
    ) -> List[Decimal]:
        """Asynchronous processing of operation chunk."""
        loop = asyncio.get_event_loop()

        # Execute in thread pool
        future = loop.run_in_executor(
            self.thread_pool, self._process_chunk_sync, chunk, method
        )

        return await future

    def _process_chunk_sync(
        self, chunk: List[tuple[ZeroSumInfiniteSet, ZeroSumInfiniteSet]], method: str
    ) -> List[Decimal]:
        """Synchronous processing of operation chunk."""
        results = []

        for set_a, set_b in chunk:
            # Check cache
            if self.use_cache and self.cache:
                cached_result = self.cache.get(
                    "zero_sum_operation", set_a.to_dict(), set_b.to_dict(), method
                )

                if cached_result is not None:
                    results.append(cached_result)
                    with self._lock:
                        self.stats["cache_hits"] += 1
                    continue

            # Execute operation
            result = set_a.zero_sum_operation(set_b, method)
            results.append(result)

            # Save to cache
            if self.use_cache and self.cache:
                self.cache.set(
                    "zero_sum_operation",
                    result,
                    set_a.to_dict(),
                    set_b.to_dict(),
                    method,
                )
                with self._lock:
                    self.stats["cache_misses"] += 1

        return results

    def parallel_compensating_sets(
        self, sets: List[ZeroSumInfiniteSet], method: str = "direct"
    ) -> List[ZeroSumInfiniteSet]:
        """Parallel search for compensating sets.

        Args:
            sets: List of sets
            method: Method for finding compensating sets

        Returns:
            List of compensating sets
        """
        # Use process pool for CPU-intensive operations
        if not self.process_pool:
            self.process_pool = Pool(processes=self.max_workers)

        # Create partial function
        find_compensating_partial = partial(
            self._find_compensating_worker, method=method
        )

        # Parallel execution
        results = self.process_pool.map(find_compensating_partial, sets)

        return results

    @staticmethod
    def _find_compensating_worker(
        infinite_set: ZeroSumInfiniteSet, method: str
    ) -> ZeroSumInfiniteSet:
        """Worker function for finding compensating set."""
        return infinite_set.find_compensating_set(method)

    async def parallel_validation(
        self, sets: List[ZeroSumInfiniteSet], tolerance: Decimal = Decimal("1e-10")
    ) -> List[Dict[str, Any]]:
        """Parallel validation of zero sums.

        Args:
            sets: List of sets for validation
            tolerance: Acceptable tolerance

        Returns:
            List of validation results
        """
        # Split into chunks
        chunks = self._create_chunks(sets, self.chunk_size)

        # Create tasks
        tasks = []
        for chunk in chunks:
            task = asyncio.create_task(self._validate_chunk_async(chunk, tolerance))
            tasks.append(task)

        # Execute and combine results
        chunk_results = await asyncio.gather(*tasks)

        results = []
        for chunk_result in chunk_results:
            results.extend(chunk_result)

        return results

    async def _validate_chunk_async(
        self, chunk: List[ZeroSumInfiniteSet], tolerance: Decimal
    ) -> List[Dict[str, Any]]:
        """Asynchronous validation of set chunk."""
        loop = asyncio.get_event_loop()

        future = loop.run_in_executor(
            self.thread_pool, self._validate_chunk_sync, chunk, tolerance
        )

        return await future

    def _validate_chunk_sync(
        self, chunk: List[ZeroSumInfiniteSet], tolerance: Decimal
    ) -> List[Dict[str, Any]]:
        """Synchronous validation of set chunk."""
        results = []

        for infinite_set in chunk:
            # Check cache
            if self.use_cache and self.cache:
                cached_result = self.cache.get(
                    "validate_zero_sum", infinite_set.id, str(tolerance)
                )

                if cached_result is not None:
                    results.append(cached_result)
                    continue

            # Execute validation
            result = infinite_set.validate_zero_sum(tolerance)
            results.append(result)

            # Save to cache
            if self.use_cache and self.cache:
                self.cache.set(
                    "validate_zero_sum", result, infinite_set.id, str(tolerance)
                )

        return results

    def parallel_convergence_analysis(
        self, sets: List[ZeroSumInfiniteSet], max_terms: int = 1000
    ) -> List[Dict[str, Any]]:
        """Parallel convergence analysis.

        Args:
            sets: List of sets for analysis
            max_terms: Maximum number of terms

        Returns:
            List of convergence analysis results
        """
        # Use process pool
        if not self.process_pool:
            self.process_pool = Pool(processes=self.max_workers)

        # Create partial function
        analyze_convergence_partial = partial(
            self._analyze_convergence_worker, max_terms=max_terms
        )

        # Parallel execution
        results = self.process_pool.map(analyze_convergence_partial, sets)

        return results

    @staticmethod
    def _analyze_convergence_worker(
        infinite_set: ZeroSumInfiniteSet, max_terms: int
    ) -> Dict[str, Any]:
        """Worker function for convergence analysis."""
        return infinite_set.convergence_analysis(max_terms)

    def batch_operations(self, operations: List[Dict[str, Any]]) -> List[Any]:
        """Batch execution of various operations.

        Args:
            operations: List of operations with parameters

        Returns:
            List of operation results
        """
        # Group operations by type
        operation_groups = {}
        for i, op in enumerate(operations):
            op_type = op["type"]
            if op_type not in operation_groups:
                operation_groups[op_type] = []
            operation_groups[op_type].append((i, op))

        # Parallel execution by groups
        futures = []
        for op_type, ops in operation_groups.items():
            future = self.thread_pool.submit(
                self._execute_operation_group, op_type, ops
            )
            futures.append(future)

        # Collect results
        all_results = {}
        for future in concurrent.futures.as_completed(futures):
            group_results = future.result()
            all_results.update(group_results)

        # Order results by original sequence
        ordered_results = [all_results[i] for i in range(len(operations))]

        return ordered_results

    def _execute_operation_group(
        self, op_type: str, operations: List[tuple[int, Dict[str, Any]]]
    ) -> Dict[int, Any]:
        """Execute a group of operations of the same type.

        Args:
            op_type: Operation type
            operations: List of operations with indices

        Returns:
            Dictionary of results with indices
        """
        results = {}

        for index, operation in operations:
            try:
                if op_type == "zero_sum_operation":
                    set_a = operation["set_a"]
                    set_b = operation["set_b"]
                    method = operation.get("method", "compensated")
                    result = set_a.zero_sum_operation(set_b, method)

                elif op_type == "find_compensating_set":
                    infinite_set = operation["set"]
                    method = operation.get("method", "direct")
                    result = infinite_set.find_compensating_set(method)

                elif op_type == "validate_zero_sum":
                    infinite_set = operation["set"]
                    tolerance = operation.get("tolerance", Decimal("1e-10"))
                    result = infinite_set.validate_zero_sum(tolerance)

                elif op_type == "convergence_analysis":
                    infinite_set = operation["set"]
                    max_terms = operation.get("max_terms", 1000)
                    result = infinite_set.convergence_analysis(max_terms)

                else:
                    raise ValueError(f"Unknown operation type: {op_type}")

                results[index] = result

            except Exception as e:
                results[index] = {"error": str(e)}

        return results

    @staticmethod
    def _create_chunks(items: List[Any], chunk_size: int) -> List[List[Any]]:
        """Split list into chunks.

        Args:
            items: List of items
            chunk_size: Chunk size

        Returns:
            List of chunks
        """
        chunks = []
        for i in range(0, len(items), chunk_size):
            chunks.append(items[i : i + chunk_size])
        return chunks

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            stats = self.stats.copy()

        # Add calculated metrics
        if stats["operations_completed"] > 0:
            stats["avg_execution_time"] = (
                stats["total_execution_time"] / stats["parallel_operations"]
            )
            stats["operations_per_second"] = (
                stats["operations_completed"] / stats["total_execution_time"]
                if stats["total_execution_time"] > 0
                else 0
            )
        else:
            stats["avg_execution_time"] = 0
            stats["operations_per_second"] = 0

        # Cache statistics
        if self.cache:
            cache_stats = self.cache.get_stats()
            stats["cache_stats"] = cache_stats

        stats["max_workers"] = self.max_workers
        stats["chunk_size"] = self.chunk_size

        return stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        with self._lock:
            self.stats = {
                "operations_completed": 0,
                "total_execution_time": 0,
                "parallel_operations": 0,
                "cache_hits": 0,
                "cache_misses": 0,
            }


# Global instance for convenience
_global_parallel_processor = None


def get_global_parallel_processor() -> ParallelTNSIM:
    """Get global instance of parallel processor."""
    global _global_parallel_processor
    if _global_parallel_processor is None:
        _global_parallel_processor = ParallelTNSIM()
    return _global_parallel_processor
