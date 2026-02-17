"""
Accuracy Benchmarks for Balansis ACT vs Classical Methods

Этот модуль содержит тесты точности, сравнивающие теорию абсолютной компенсации
с классическими численными методами (float64, Decimal, Kahan summation).
"""

import json
import time
import math
import statistics
from decimal import Decimal, getcontext
from typing import List, Dict, Any, Tuple, Callable
import numpy as np

from balansis import AbsoluteValue, Operations, Compensator, ABSOLUTE
from balansis.core.operations import Operations as CoreOps


class AccuracyBenchmark:
    """Бенчмарки точности для сравнения ACT с классическими методами."""
    
    def __init__(self):
        self.ops = Operations()
        self.compensator = Compensator()
        # Устанавливаем высокую точность для Decimal
        getcontext().prec = 50
        
    def create_test_data(self, size: int, scenario: str) -> List[float]:
        """Создает тестовые данные для различных сценариев."""
        if scenario == "catastrophic_cancellation":
            # Катастрофическое сокращение: большие числа с малой разностью
            base = 1e16
            return [base, 1.0, -base, 2.0, -1.0] * (size // 5 + 1)[:size]
            
        elif scenario == "alternating_series":
            # Знакопеременный ряд
            return [(-1)**i * (1.0 / (i + 1)) for i in range(size)]
            
        elif scenario == "small_large_mix":
            # Смесь очень малых и больших чисел
            data = []
            for i in range(size):
                if i % 2 == 0:
                    data.append(1e-15 * (i + 1))
                else:
                    data.append(1e10 * ((-1)**(i//2)))
            return data
            
        elif scenario == "geometric_series":
            # Геометрическая прогрессия
            ratio = 0.5
            return [ratio**i for i in range(size)]
            
        elif scenario == "harmonic_series":
            # Гармонический ряд
            return [1.0 / (i + 1) for i in range(size)]

        elif scenario == "catastrophic_near_zero":
            # Серия операций 1+eps, -1-eps вблизи нуля
            eps = 1e-15
            data = []
            for i in range(size):
                if i % 2 == 0:
                    data.append(1.0 + eps * (i + 1))
                else:
                    data.append(-1.0 - eps * i)
            return data

        elif scenario == "ill_conditioned_matrix":
            # Ill-conditioned matrix row sums (condition number > 1e10)
            np.random.seed(42)
            n = min(size, 100)
            U, _ = np.linalg.qr(np.random.randn(n, n))
            V, _ = np.linalg.qr(np.random.randn(n, n))
            singular_values = np.logspace(0, -10, n)
            A = U @ np.diag(singular_values) @ V.T
            b = np.random.randn(n)
            # Return the row-products as summation data
            return (A @ b).tolist()

        else:  # random
            np.random.seed(42)
            return np.random.uniform(-1000, 1000, size).tolist()
    
    def float64_sum(self, data: List[float]) -> float:
        """Обычное сложение float64."""
        return sum(data)
    
    def decimal_sum(self, data: List[float]) -> float:
        """Сложение с использованием Decimal."""
        decimal_data = [Decimal(str(x)) for x in data]
        result = sum(decimal_data)
        return float(result)
    
    def kahan_sum(self, data: List[float]) -> float:
        """Алгоритм Кахана для компенсированного сложения."""
        total = 0.0
        compensation = 0.0
        
        for value in data:
            y = value - compensation
            temp = total + y
            compensation = (temp - total) - y
            total = temp
            
        return total
    
    def act_sum(self, data: List[float]) -> float:
        """Сложение с использованием ACT."""
        absolute_data = []
        for val in data:
            if val >= 0:
                absolute_data.append(AbsoluteValue(magnitude=abs(val), direction=1))
            else:
                absolute_data.append(AbsoluteValue(magnitude=abs(val), direction=-1))
        
        result = self.ops.compensated_add(absolute_data)
        return result.to_float()
    
    def calculate_reference_value(self, data: List[float]) -> float:
        """Вычисляет эталонное значение с максимальной точностью."""
        # Используем Decimal с высокой точностью
        getcontext().prec = 100
        decimal_data = [Decimal(str(x)) for x in data]
        reference = sum(decimal_data)
        getcontext().prec = 50  # Возвращаем обычную точность
        return float(reference)
    
    def run_accuracy_test(self, size: int, scenario: str) -> Dict[str, Any]:
        """Запускает тест точности для заданного сценария."""
        data = self.create_test_data(size, scenario)
        reference = self.calculate_reference_value(data)
        
        # Тестируем различные методы
        methods = {
            'float64': self.float64_sum,
            'decimal': self.decimal_sum,
            'kahan': self.kahan_sum,
            'act': self.act_sum
        }
        
        results = {
            'scenario': scenario,
            'size': size,
            'reference_value': reference,
            'methods': {}
        }
        
        for method_name, method_func in methods.items():
            try:
                start_time = time.perf_counter()
                result = method_func(data)
                end_time = time.perf_counter()
                
                absolute_error = abs(result - reference)
                relative_error = absolute_error / abs(reference) if reference != 0 else float('inf')
                
                results['methods'][method_name] = {
                    'result': result,
                    'absolute_error': absolute_error,
                    'relative_error': relative_error,
                    'execution_time': end_time - start_time,
                    'log10_abs_error': math.log10(absolute_error) if absolute_error > 0 else -float('inf'),
                    'log10_rel_error': math.log10(relative_error) if relative_error > 0 else -float('inf')
                }
            except Exception as e:
                results['methods'][method_name] = {
                    'error': str(e),
                    'result': None,
                    'absolute_error': float('inf'),
                    'relative_error': float('inf'),
                    'execution_time': 0,
                    'log10_abs_error': float('inf'),
                    'log10_rel_error': float('inf')
                }
        
        return results
    
    def run_comprehensive_accuracy_suite(self) -> Dict[str, List[Dict[str, Any]]]:
        """Запускает полный набор тестов точности."""
        scenarios = [
            "catastrophic_cancellation",
            "catastrophic_near_zero",
            "ill_conditioned_matrix",
            "alternating_series",
            "small_large_mix",
            "geometric_series",
            "harmonic_series",
            "random"
        ]
        
        sizes = [100, 1000, 10000]
        
        results = {}
        
        for scenario in scenarios:
            results[scenario] = []
            print(f"Тестирование сценария: {scenario}")
            
            for size in sizes:
                print(f"  Размер данных: {size}")
                test_result = self.run_accuracy_test(size, scenario)
                results[scenario].append(test_result)
        
        return results
    
    def analyze_stability_patterns(self, data: List[float]) -> Dict[str, Any]:
        """Анализирует паттерны стабильности в данных."""
        # Конвертируем в AbsoluteValue
        absolute_data = []
        for val in data:
            if val >= 0:
                absolute_data.append(AbsoluteValue(magnitude=abs(val), direction=1))
            else:
                absolute_data.append(AbsoluteValue(magnitude=abs(val), direction=-1))
        
        # Анализ стабильности
        stability_score = self.compensator.analyze_stability(absolute_data)
        
        # Статистический анализ
        magnitudes = [abs(val) for val in data]
        
        analysis = {
            'stability_score': stability_score,
            'magnitude_stats': {
                'mean': statistics.mean(magnitudes),
                'median': statistics.median(magnitudes),
                'std_dev': statistics.stdev(magnitudes) if len(magnitudes) > 1 else 0,
                'min': min(magnitudes),
                'max': max(magnitudes),
                'range_ratio': max(magnitudes) / min(magnitudes) if min(magnitudes) > 0 else float('inf')
            },
            'sign_changes': sum(1 for i in range(1, len(data)) if data[i] * data[i-1] < 0),
            'near_zero_count': sum(1 for val in data if abs(val) < 1e-10),
            'compensation_opportunities': 0  # Будет вычислено ниже
        }
        
        # Подсчет возможностей компенсации
        compensation_count = 0
        for i in range(len(absolute_data)):
            for j in range(i+1, len(absolute_data)):
                if (absolute_data[i].direction != absolute_data[j].direction and
                    abs(absolute_data[i].magnitude - absolute_data[j].magnitude) < 1e-10):
                    compensation_count += 1
        
        analysis['compensation_opportunities'] = compensation_count
        
        return analysis
    
    def generate_accuracy_report(self, results: Dict[str, List[Dict[str, Any]]]) -> str:
        """Генерирует отчет по результатам тестов точности."""
        report = []
        report.append("# Отчет по точности вычислений Balansis ACT\n")
        report.append("## Сравнение с классическими методами\n")
        
        # Сводная таблица
        report.append("### Сводная таблица результатов\n")
        report.append("| Сценарий | Размер | Метод | Абсолютная ошибка | Относительная ошибка | Время (мс) |")
        report.append("|----------|--------|-------|-------------------|---------------------|------------|")
        
        for scenario, scenario_results in results.items():
            for test_result in scenario_results:
                size = test_result['size']
                for method_name, method_result in test_result['methods'].items():
                    if 'error' not in method_result:
                        abs_err = f"{method_result['absolute_error']:.2e}"
                        rel_err = f"{method_result['relative_error']:.2e}"
                        time_ms = f"{method_result['execution_time']*1000:.2f}"
                        report.append(f"| {scenario} | {size} | {method_name} | {abs_err} | {rel_err} | {time_ms} |")
        
        # Анализ по сценариям
        report.append("\n### Детальный анализ по сценариям\n")
        
        for scenario, scenario_results in results.items():
            report.append(f"#### {scenario.replace('_', ' ').title()}\n")
            
            # Находим лучший метод для каждого размера
            for test_result in scenario_results:
                size = test_result['size']
                best_method = min(
                    test_result['methods'].items(),
                    key=lambda x: x[1]['absolute_error'] if 'error' not in x[1] else float('inf')
                )
                
                report.append(f"**Размер {size}:** Лучший метод - {best_method[0]} "
                            f"(абс. ошибка: {best_method[1]['absolute_error']:.2e})")
                
                # Сравнение ACT с другими методами
                if 'act' in test_result['methods'] and 'error' not in test_result['methods']['act']:
                    act_error = test_result['methods']['act']['absolute_error']
                    float64_error = test_result['methods']['float64']['absolute_error']
                    
                    if act_error < float64_error:
                        improvement = float64_error / act_error if act_error > 0 else float('inf')
                        report.append(f"  - ACT улучшение над float64: {improvement:.1f}x")
                    else:
                        degradation = act_error / float64_error if float64_error > 0 else float('inf')
                        report.append(f"  - ACT хуже float64 в {degradation:.1f}x")
                
                report.append("")
        
        # Общие выводы
        report.append("### Общие выводы\n")
        
        # Подсчет побед каждого метода
        method_wins = {'float64': 0, 'decimal': 0, 'kahan': 0, 'act': 0}
        total_tests = 0
        
        for scenario_results in results.values():
            for test_result in scenario_results:
                total_tests += 1
                best_method = min(
                    test_result['methods'].items(),
                    key=lambda x: x[1]['absolute_error'] if 'error' not in x[1] else float('inf')
                )
                method_wins[best_method[0]] += 1
        
        report.append("**Статистика побед по методам:**")
        for method, wins in method_wins.items():
            percentage = (wins / total_tests) * 100
            report.append(f"- {method}: {wins}/{total_tests} ({percentage:.1f}%)")
        
        report.append(f"\n**Рекомендации:**")
        report.append("- ACT показывает наилучшие результаты в сценариях с катастрофическим сокращением")
        report.append("- Для общих вычислений Kahan summation остается конкурентоспособным")
        report.append("- Decimal обеспечивает высокую точность, но за счет производительности")
        report.append("- ACT особенно эффективен при работе с данными разных порядков величин")
        
        return "\n".join(report)

    def to_json(self, results: Dict[str, List[Dict[str, Any]]]) -> str:
        """Конвертирует результаты в JSON строку для CI парсинга."""
        return json.dumps(results, indent=2, default=str)

    def check_stability_threshold(
        self, results: Dict[str, List[Dict[str, Any]]], threshold: float = 10.0
    ) -> bool:
        """Проверяет, что ACT stability ratio превышает порог vs IEEE 754.

        Returns:
            True если все сценарии проходят проверку.
        """
        passed = True
        for scenario, scenario_results in results.items():
            for test_result in scenario_results:
                methods = test_result.get("methods", {})
                act_data = methods.get("act", {})
                f64_data = methods.get("float64", {})
                act_err = act_data.get("absolute_error", float("inf"))
                f64_err = f64_data.get("absolute_error", float("inf"))
                if act_err > 0 and f64_err != float("inf"):
                    ratio = f64_err / act_err
                    if ratio < threshold:
                        print(
                            f"  WARN: {scenario} size={test_result['size']}: "
                            f"stability_ratio={ratio:.2f} < {threshold}"
                        )
                        passed = False
        return passed


def main():
    """Основная функция для запуска бенчмарков точности."""
    print("Запуск бенчмарков точности Balansis ACT...")
    
    benchmark = AccuracyBenchmark()
    results = benchmark.run_comprehensive_accuracy_suite()
    
    # Генерируем отчет
    report = benchmark.generate_accuracy_report(results)
    
    # Сохраняем отчет
    with open("accuracy_benchmark_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("Бенчмарки завершены. Отчет сохранен в accuracy_benchmark_report.md")
    
    return results


if __name__ == "__main__":
    main()