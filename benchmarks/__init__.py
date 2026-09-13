"""Benchmark entry points; optional plotting dependencies load on demand."""

from importlib import import_module

_MODULES = {
    "AccuracyBenchmark": "accuracy_benchmarks",
    "PerformanceBenchmark": "performance_benchmarks",
    "LinalgBenchmark": "linalg_benchmarks",
    "MLBenchmark": "ml_benchmarks",
    "RegressionTracker": "regression_tracker",
    "BenchmarkVisualizer": "visualization",
}
__all__ = list(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
    globals()[name] = value
    return value
