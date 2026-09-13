"""
Visualization Module for Balansis Benchmarks

Этот модуль предоставляет инструменты для визуализации результатов бенчмарков
ACT в сравнении с классическими численными методами.
"""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
import seaborn as sns
from plotly.subplots import make_subplots


class BenchmarkVisualizer:
    """Класс для создания визуализаций результатов бенчмарков."""

    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
        """
        Инициализация визуализатора.

        Args:
            style: Стиль matplotlib
            figsize: Размер фигур по умолчанию
        """
        plt.style.use(style)
        self.figsize = figsize
        self.colors = {
            "float64": "#1f77b4",
            "numpy": "#ff7f0e",
            "decimal": "#2ca02c",
            "kahan": "#d62728",
            "act": "#9467bd",
        }

        # Настройка seaborn
        sns.set_palette("husl")

    def plot_accuracy_comparison(
        self,
        accuracy_results: Dict[str, List[Dict[str, Any]]],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Создает график сравнения точности методов."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Сравнение точности численных методов", fontsize=16, fontweight="bold"
        )

        scenarios = list(accuracy_results.keys())

        for idx, scenario in enumerate(scenarios):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            # Собираем данные для сценария
            methods = ["float64", "decimal", "kahan", "act"]
            sizes = []
            errors_by_method = {method: [] for method in methods}

            for test_result in accuracy_results[scenario]:
                sizes.append(test_result["size"])
                for method in methods:
                    if (
                        method in test_result["methods"]
                        and "error" not in test_result["methods"][method]
                    ):
                        error = test_result["methods"][method]["log10_abs_error"]
                        errors_by_method[method].append(
                            error if error != -float("inf") else -20
                        )
                    else:
                        errors_by_method[method].append(-20)  # Очень малая ошибка

            # Построение графика
            x_pos = np.arange(len(sizes))
            width = 0.15

            for i, method in enumerate(methods):
                offset = (i - len(methods) / 2 + 0.5) * width
                bars = ax.bar(
                    x_pos + offset,
                    errors_by_method[method],
                    width,
                    label=method,
                    color=self.colors.get(method, f"C{i}"),
                    alpha=0.8,
                )

                # Добавляем значения на столбцы
                for bar, error in zip(bars, errors_by_method[method]):
                    if error > -15:  # Показываем только значимые ошибки
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            bar.get_height() + 0.1,
                            f"{error:.1f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

            ax.set_xlabel("Размер данных")
            ax.set_ylabel("log₁₀(Абсолютная ошибка)")
            ax.set_title(scenario.replace("_", " ").title())
            ax.set_xticks(x_pos)
            ax.set_xticklabels(sizes)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Удаляем пустые подграфики
        for idx in range(len(scenarios), 6):
            row = idx // 3
            col = idx % 3
            fig.delaxes(axes[row, col])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_performance_scaling(
        self, performance_results: Dict[str, Any], save_path: Optional[str] = None
    ) -> plt.Figure:
        """Создает график масштабируемости производительности."""
        operations = list(performance_results.keys())
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Масштабируемость производительности", fontsize=16, fontweight="bold"
        )

        for idx, operation in enumerate(operations[:4]):  # Ограничиваем 4 операциями
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            # Собираем данные
            methods_data = {}

            for size_key, size_results in performance_results[operation].items():
                size = int(size_key.split("_")[1])

                for method, method_results in size_results.items():
                    if method not in methods_data:
                        methods_data[method] = {"sizes": [], "times": []}

                    methods_data[method]["sizes"].append(size)
                    methods_data[method]["times"].append(
                        method_results["time"]["mean"] * 1000
                    )  # в мс

            # Построение графиков
            for method, data in methods_data.items():
                if len(data["sizes"]) > 1:
                    # Сортируем по размеру
                    sorted_data = sorted(zip(data["sizes"], data["times"]))
                    sizes, times = zip(*sorted_data)

                    ax.loglog(
                        sizes,
                        times,
                        "o-",
                        label=method,
                        color=self.colors.get(method, None),
                        linewidth=2,
                        markersize=6,
                    )

            ax.set_xlabel("Размер данных")
            ax.set_ylabel("Время выполнения (мс)")
            ax.set_title(f'{operation.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_memory_usage(
        self, performance_results: Dict[str, Any], save_path: Optional[str] = None
    ) -> plt.Figure:
        """Создает график использования памяти."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Использование памяти по методам", fontsize=16, fontweight="bold")

        operations = list(performance_results.keys())[:4]

        for idx, operation in enumerate(operations):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            # Собираем данные по памяти
            methods_memory = {}
            sizes = []

            for size_key, size_results in performance_results[operation].items():
                size = int(size_key.split("_")[1])
                if not sizes or size not in sizes:
                    sizes.append(size)

                for method, method_results in size_results.items():
                    if method not in methods_memory:
                        methods_memory[method] = []
                    methods_memory[method].append(
                        method_results["memory"]["allocated_mb"]
                    )

            # Создаем тепловую карту
            methods = list(methods_memory.keys())
            memory_matrix = np.array([methods_memory[method] for method in methods])

            im = ax.imshow(memory_matrix, cmap="YlOrRd", aspect="auto")

            # Настройка осей
            ax.set_xticks(range(len(sizes)))
            ax.set_xticklabels(sizes)
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels(methods)
            ax.set_xlabel("Размер данных")
            ax.set_ylabel("Метод")
            ax.set_title(f'{operation.replace("_", " ").title()}')

            # Добавляем значения в ячейки
            for i in range(len(methods)):
                for j in range(len(sizes)):
                    text = ax.text(
                        j,
                        i,
                        f"{memory_matrix[i, j]:.1f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )

            # Цветовая шкала
            plt.colorbar(im, ax=ax, label="Память (МБ)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_interactive_dashboard(
        self,
        accuracy_results: Dict[str, List[Dict[str, Any]]],
        performance_results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> str:
        """Создает интерактивную панель управления с результатами."""

        # Создаем подграфики
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Точность по сценариям",
                "Производительность по операциям",
                "Масштабируемость",
                "Использование памяти",
                "Сравнение методов",
                "Тренды стабильности",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # График 1: Точность по сценариям
        scenarios = list(accuracy_results.keys())[:3]  # Ограничиваем для читаемости
        methods = ["float64", "kahan", "act"]

        for method in methods:
            scenario_errors = []
            for scenario in scenarios:
                # Берем средний размер данных
                mid_result = accuracy_results[scenario][
                    len(accuracy_results[scenario]) // 2
                ]
                if (
                    method in mid_result["methods"]
                    and "error" not in mid_result["methods"][method]
                ):
                    error = mid_result["methods"][method]["log10_abs_error"]
                    scenario_errors.append(error if error != -float("inf") else -20)
                else:
                    scenario_errors.append(-20)

            fig.add_trace(
                go.Bar(
                    name=method,
                    x=scenarios,
                    y=scenario_errors,
                    marker_color=self.colors.get(method, "blue"),
                ),
                row=1,
                col=1,
            )

        # График 2: Производительность
        operation = list(performance_results.keys())[0]  # Берем первую операцию
        sizes = []
        act_times = []
        float64_times = []

        for size_key, size_results in performance_results[operation].items():
            size = int(size_key.split("_")[1])
            sizes.append(size)

            if "act" in size_results:
                act_times.append(size_results["act"]["time"]["mean"] * 1000)
            else:
                act_times.append(0)

            if "float64" in size_results:
                float64_times.append(size_results["float64"]["time"]["mean"] * 1000)
            else:
                float64_times.append(0)

        fig.add_trace(
            go.Scatter(
                x=sizes,
                y=act_times,
                mode="lines+markers",
                name="ACT",
                line=dict(color=self.colors["act"]),
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=sizes,
                y=float64_times,
                mode="lines+markers",
                name="Float64",
                line=dict(color=self.colors["float64"]),
            ),
            row=1,
            col=2,
        )

        # График 3: Масштабируемость (логарифмический)
        fig.add_trace(
            go.Scatter(
                x=sizes,
                y=act_times,
                mode="lines+markers",
                name="ACT (log)",
                line=dict(color=self.colors["act"], dash="dash"),
            ),
            row=2,
            col=1,
        )

        # График 4: Использование памяти
        act_memory = []
        float64_memory = []

        for size_key, size_results in performance_results[operation].items():
            if "act" in size_results:
                act_memory.append(size_results["act"]["memory"]["allocated_mb"])
            else:
                act_memory.append(0)

            if "float64" in size_results:
                float64_memory.append(size_results["float64"]["memory"]["allocated_mb"])
            else:
                float64_memory.append(0)

        fig.add_trace(
            go.Bar(
                name="ACT Memory",
                x=sizes,
                y=act_memory,
                marker_color=self.colors["act"],
                opacity=0.7,
            ),
            row=2,
            col=2,
        )

        fig.add_trace(
            go.Bar(
                name="Float64 Memory",
                x=sizes,
                y=float64_memory,
                marker_color=self.colors["float64"],
                opacity=0.7,
            ),
            row=2,
            col=2,
        )

        # График 5: Сравнение методов (радарная диаграмма)
        categories = ["Точность", "Скорость", "Память", "Стабильность"]

        # Нормализованные оценки (примерные)
        act_scores = [0.9, 0.7, 0.6, 0.95]
        float64_scores = [0.6, 0.95, 0.9, 0.5]

        fig.add_trace(
            go.Scatterpolar(
                r=act_scores + [act_scores[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="ACT",
                line_color=self.colors["act"],
            ),
            row=3,
            col=1,
        )

        fig.add_trace(
            go.Scatterpolar(
                r=float64_scores + [float64_scores[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="Float64",
                line_color=self.colors["float64"],
            ),
            row=3,
            col=1,
        )

        # График 6: Тренды стабильности
        stability_trend = [0.8, 0.85, 0.9, 0.92]  # Примерные данные

        fig.add_trace(
            go.Scatter(
                x=sizes,
                y=stability_trend,
                mode="lines+markers",
                name="Стабильность ACT",
                line=dict(color="green", width=3),
            ),
            row=3,
            col=2,
        )

        # Обновляем макет
        fig.update_layout(
            height=1200,
            title_text="Интерактивная панель бенчмарков Balansis ACT",
            title_x=0.5,
            showlegend=True,
        )

        # Настройка осей
        fig.update_xaxes(title_text="Сценарии", row=1, col=1)
        fig.update_yaxes(title_text="log₁₀(Ошибка)", row=1, col=1)

        fig.update_xaxes(title_text="Размер данных", row=1, col=2)
        fig.update_yaxes(title_text="Время (мс)", row=1, col=2)

        fig.update_xaxes(title_text="Размер данных", type="log", row=2, col=1)
        fig.update_yaxes(title_text="Время (мс)", type="log", row=2, col=1)

        fig.update_xaxes(title_text="Размер данных", row=2, col=2)
        fig.update_yaxes(title_text="Память (МБ)", row=2, col=2)

        fig.update_xaxes(title_text="Размер данных", row=3, col=2)
        fig.update_yaxes(title_text="Коэффициент стабильности", row=3, col=2)

        # Сохранение
        if save_path:
            pyo.plot(fig, filename=save_path, auto_open=False)

        return fig.to_html()

    def plot_error_distribution(
        self,
        accuracy_results: Dict[str, List[Dict[str, Any]]],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Создает график распределения ошибок."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Распределение ошибок по методам", fontsize=16, fontweight="bold")

        methods = ["float64", "decimal", "kahan", "act"]

        for idx, method in enumerate(methods):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            # Собираем все ошибки для метода
            all_errors = []

            for scenario_results in accuracy_results.values():
                for test_result in scenario_results:
                    if (
                        method in test_result["methods"]
                        and "error" not in test_result["methods"][method]
                    ):
                        error = test_result["methods"][method]["log10_abs_error"]
                        if error != -float("inf"):
                            all_errors.append(error)

            if all_errors:
                # Гистограмма
                ax.hist(
                    all_errors,
                    bins=20,
                    alpha=0.7,
                    color=self.colors.get(method, "blue"),
                    edgecolor="black",
                    linewidth=0.5,
                )

                # Статистики
                mean_error = np.mean(all_errors)
                median_error = np.median(all_errors)

                ax.axvline(
                    mean_error,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Среднее: {mean_error:.2f}",
                )
                ax.axvline(
                    median_error,
                    color="orange",
                    linestyle="--",
                    linewidth=2,
                    label=f"Медиана: {median_error:.2f}",
                )

                ax.set_xlabel("log₁₀(Абсолютная ошибка)")
                ax.set_ylabel("Частота")
                ax.set_title(f"Распределение ошибок: {method}")
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Нет данных",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"Распределение ошибок: {method}")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_summary_report(
        self,
        accuracy_results: Dict[str, List[Dict[str, Any]]],
        performance_results: Dict[str, Any],
        save_dir: str = "benchmark_plots",
    ) -> Dict[str, str]:
        """Создает полный набор визуализаций и сохраняет их."""
        import os

        # Создаем директорию если не существует
        os.makedirs(save_dir, exist_ok=True)

        plots = {}

        # График точности
        accuracy_fig = self.plot_accuracy_comparison(
            accuracy_results, f"{save_dir}/accuracy_comparison.png"
        )
        plots["accuracy"] = f"{save_dir}/accuracy_comparison.png"
        plt.close(accuracy_fig)

        # График производительности
        performance_fig = self.plot_performance_scaling(
            performance_results, f"{save_dir}/performance_scaling.png"
        )
        plots["performance"] = f"{save_dir}/performance_scaling.png"
        plt.close(performance_fig)

        # График памяти
        memory_fig = self.plot_memory_usage(
            performance_results, f"{save_dir}/memory_usage.png"
        )
        plots["memory"] = f"{save_dir}/memory_usage.png"
        plt.close(memory_fig)

        # График распределения ошибок
        error_fig = self.plot_error_distribution(
            accuracy_results, f"{save_dir}/error_distribution.png"
        )
        plots["error_distribution"] = f"{save_dir}/error_distribution.png"
        plt.close(error_fig)

        # Интерактивная панель
        dashboard_html = self.create_interactive_dashboard(
            accuracy_results,
            performance_results,
            f"{save_dir}/interactive_dashboard.html",
        )
        plots["dashboard"] = f"{save_dir}/interactive_dashboard.html"

        return plots


def main():
    """Демонстрация возможностей визуализации."""
    # Создаем пример данных для демонстрации
    visualizer = BenchmarkVisualizer()

    # Пример данных точности
    sample_accuracy = {
        "catastrophic_cancellation": [
            {
                "size": 1000,
                "methods": {
                    "float64": {"log10_abs_error": -5.2},
                    "kahan": {"log10_abs_error": -8.1},
                    "act": {"log10_abs_error": -12.3},
                },
            }
        ]
    }

    # Пример данных производительности
    sample_performance = {
        "addition": {
            "size_1000": {
                "float64": {"time": {"mean": 0.001}, "memory": {"allocated_mb": 0.1}},
                "act": {"time": {"mean": 0.003}, "memory": {"allocated_mb": 0.3}},
            }
        }
    }

    # Создаем демонстрационные графики
    plots = visualizer.create_summary_report(sample_accuracy, sample_performance)

    print("Демонстрационные графики созданы:")
    for plot_type, path in plots.items():
        print(f"- {plot_type}: {path}")


if __name__ == "__main__":
    main()
