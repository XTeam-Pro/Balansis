"""
Balansis Benchmarks Package

Этот пакет содержит бенчмарки для сравнения производительности и точности
теории абсолютной компенсации (ACT) с классическими численными методами.

Модули:
- accuracy_benchmarks: Тесты точности вычислений
- performance_benchmarks: Тесты производительности
- stability_benchmarks: Тесты стабильности
- linalg_benchmarks: Бенчмарки линейной алгебры (GEMM, SVD, QR)
- ml_benchmarks: Бенчмарки ML оптимизаторов
- regression_tracker: Отслеживание регрессий
- visualization: Визуализация результатов
- utils: Вспомогательные функции
"""

from .accuracy_benchmarks import AccuracyBenchmark
from .performance_benchmarks import PerformanceBenchmark
from .stability_benchmarks import StabilityBenchmark
from .linalg_benchmarks import LinalgBenchmark
from .ml_benchmarks import MLBenchmark
from .regression_tracker import RegressionTracker
from .visualization import BenchmarkVisualizer
from .utils import BenchmarkUtils

__all__ = [
    'AccuracyBenchmark',
    'PerformanceBenchmark',
    'StabilityBenchmark',
    'LinalgBenchmark',
    'MLBenchmark',
    'RegressionTracker',
    'BenchmarkVisualizer',
    'BenchmarkUtils'
]

__version__ = "0.2.0"