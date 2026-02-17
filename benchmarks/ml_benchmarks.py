"""
ML Benchmarks for Balansis ACT Optimizers

Compares EternalOptimizer (ACT) vs Adam on standard optimization
test functions: Rosenbrock and Rastrigin.
"""

import json
import math
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from balansis.core.absolute import AbsoluteValue
from balansis.core.eternity import EternalRatio
from balansis.core.operations import Operations


class _SimpleParam:
    """Lightweight parameter wrapper mimicking torch.Tensor for non-torch usage."""

    def __init__(self, data: np.ndarray):
        self.data = data.copy()
        self.grad: Optional[np.ndarray] = None


class _NumpyEternalOptimizer:
    """Pure-numpy EternalOptimizer for benchmarking without torch dependency."""

    def __init__(self, params: List[_SimpleParam], lr: float = 1e-3,
                 momentum: float = 0.0, weight_decay: float = 0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.state: Dict[int, Dict[str, Any]] = {}

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            g = p.grad.copy()
            pid = id(p)
            if pid not in self.state:
                self.state[pid] = {"momentum_buffer": None, "step": 0}
            state = self.state[pid]
            state["step"] += 1

            # Weight decay
            if self.weight_decay != 0.0:
                decay = max(1.0 - self.lr * self.weight_decay, 1e-15)
                p.data *= decay

            # Momentum
            if self.momentum != 0.0:
                if state["momentum_buffer"] is None:
                    state["momentum_buffer"] = g.copy()
                else:
                    state["momentum_buffer"] = (
                        self.momentum * state["momentum_buffer"]
                        + (1.0 - self.momentum) * g
                    )
                g = state["momentum_buffer"]

            # ACT-scaled learning rate
            grad_norm = float(np.linalg.norm(g))
            num = AbsoluteValue.from_float(self.lr)
            den = (AbsoluteValue.from_float(grad_norm)
                   if grad_norm > 0 else AbsoluteValue.unit_positive())
            ratio = EternalRatio(numerator=num, denominator=den)
            scaled_lr = ratio.numerical_value()

            p.data -= scaled_lr * g


class _NumpyAdamOptimizer:
    """Pure-numpy Adam optimizer for fair comparison."""

    def __init__(self, params: List[_SimpleParam], lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.state: Dict[int, Dict[str, Any]] = {}

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            g = p.grad
            pid = id(p)
            if pid not in self.state:
                self.state[pid] = {
                    "m": np.zeros_like(p.data),
                    "v": np.zeros_like(p.data),
                    "step": 0,
                }
            state = self.state[pid]
            state["step"] += 1
            t = state["step"]

            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * g
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * g ** 2

            m_hat = state["m"] / (1 - self.beta1 ** t)
            v_hat = state["v"] / (1 - self.beta2 ** t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def rosenbrock(x: np.ndarray) -> Tuple[float, np.ndarray]:
    """Rosenbrock function and gradient. Minimum at (1, 1, ..., 1)."""
    n = len(x)
    val = sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
              for i in range(n - 1))
    grad = np.zeros(n)
    for i in range(n - 1):
        grad[i] += -400.0 * x[i] * (x[i + 1] - x[i] ** 2) - 2.0 * (1 - x[i])
        grad[i + 1] += 200.0 * (x[i + 1] - x[i] ** 2)
    return float(val), grad


def rastrigin(x: np.ndarray) -> Tuple[float, np.ndarray]:
    """Rastrigin function and gradient. Minimum at (0, 0, ..., 0)."""
    A = 10.0
    n = len(x)
    val = A * n + sum(xi ** 2 - A * math.cos(2 * math.pi * xi) for xi in x)
    grad = np.array([
        2.0 * xi + A * 2 * math.pi * math.sin(2 * math.pi * xi)
        for xi in x
    ])
    return float(val), grad


class MLBenchmark:
    """Benchmarks comparing EternalOptimizer vs Adam on optimization problems."""

    def __init__(self, max_steps: int = 2000, dim: int = 5):
        self.max_steps = max_steps
        self.dim = dim

    def _run_optimizer(self, opt_class, opt_kwargs, func, x0, max_steps):
        """Run an optimizer on a function and track convergence."""
        param = _SimpleParam(x0.copy())
        opt = opt_class([param], **opt_kwargs)

        history = []
        grad_norms = []
        t0 = time.perf_counter()

        for step in range(max_steps):
            val, grad = func(param.data)
            param.grad = grad
            opt.step()

            gn = float(np.linalg.norm(grad))
            history.append(val)
            grad_norms.append(gn)

            if gn < 1e-8:
                break

        elapsed = time.perf_counter() - t0
        return {
            "final_value": history[-1],
            "final_grad_norm": grad_norms[-1],
            "steps": len(history),
            "time_s": elapsed,
            "converged": grad_norms[-1] < 1e-4,
            "history_sample": history[::max(1, len(history) // 20)],
        }

    def benchmark_rosenbrock(self) -> Dict[str, Any]:
        """Compare optimizers on Rosenbrock function."""
        np.random.seed(42)
        x0 = np.random.randn(self.dim) * 0.5

        eternal_result = self._run_optimizer(
            _NumpyEternalOptimizer,
            {"lr": 1e-3, "momentum": 0.9},
            rosenbrock, x0, self.max_steps,
        )

        adam_result = self._run_optimizer(
            _NumpyAdamOptimizer,
            {"lr": 1e-3},
            rosenbrock, x0, self.max_steps,
        )

        return {
            "function": "rosenbrock",
            "dimension": self.dim,
            "eternal_optimizer": eternal_result,
            "adam_optimizer": adam_result,
        }

    def benchmark_rastrigin(self) -> Dict[str, Any]:
        """Compare optimizers on Rastrigin function."""
        np.random.seed(42)
        x0 = np.random.randn(self.dim) * 0.5

        eternal_result = self._run_optimizer(
            _NumpyEternalOptimizer,
            {"lr": 1e-3, "momentum": 0.9},
            rastrigin, x0, self.max_steps,
        )

        adam_result = self._run_optimizer(
            _NumpyAdamOptimizer,
            {"lr": 1e-3},
            rastrigin, x0, self.max_steps,
        )

        return {
            "function": "rastrigin",
            "dimension": self.dim,
            "eternal_optimizer": eternal_result,
            "adam_optimizer": adam_result,
        }

    def benchmark_gradient_stability(self) -> Dict[str, Any]:
        """Compare gradient stability across many steps on Rosenbrock."""
        np.random.seed(42)
        x0 = np.random.randn(self.dim) * 0.5

        # Track gradient norms for stability analysis
        def run_with_grad_tracking(opt_class, opt_kwargs):
            param = _SimpleParam(x0.copy())
            opt = opt_class([param], **opt_kwargs)
            grad_norms = []
            for _ in range(self.max_steps):
                _, grad = rosenbrock(param.data)
                param.grad = grad
                opt.step()
                grad_norms.append(float(np.linalg.norm(grad)))
            arr = np.array(grad_norms)
            return {
                "mean_grad_norm": float(np.mean(arr)),
                "std_grad_norm": float(np.std(arr)),
                "max_grad_norm": float(np.max(arr)),
                "min_grad_norm": float(np.min(arr)),
                "grad_norm_ratio_max_min": float(np.max(arr) / np.min(arr)) if np.min(arr) > 0 else float("inf"),
                "num_exploding": int(np.sum(arr > 1e6)),
                "num_vanishing": int(np.sum(arr < 1e-10)),
            }

        eternal_stability = run_with_grad_tracking(
            _NumpyEternalOptimizer, {"lr": 1e-3, "momentum": 0.9})
        adam_stability = run_with_grad_tracking(
            _NumpyAdamOptimizer, {"lr": 1e-3})

        return {
            "analysis": "gradient_stability",
            "dimension": self.dim,
            "max_steps": self.max_steps,
            "eternal_optimizer": eternal_stability,
            "adam_optimizer": adam_stability,
        }

    def run_all(self) -> Dict[str, Any]:
        """Run all ML benchmarks."""
        print("  ML benchmark: Rosenbrock...")
        rosenbrock_result = self.benchmark_rosenbrock()
        print("  ML benchmark: Rastrigin...")
        rastrigin_result = self.benchmark_rastrigin()
        print("  ML benchmark: Gradient stability...")
        stability_result = self.benchmark_gradient_stability()

        return {
            "rosenbrock": rosenbrock_result,
            "rastrigin": rastrigin_result,
            "gradient_stability": stability_result,
        }

    def to_json(self) -> str:
        """Run benchmarks and return JSON string."""
        results = self.run_all()
        return json.dumps(results, indent=2, default=str)


def main():
    print("Running ML benchmarks...")
    bench = MLBenchmark()
    results = bench.run_all()

    with open("ml_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nResults saved to ml_benchmark_results.json")

    for func_name in ["rosenbrock", "rastrigin"]:
        r = results[func_name]
        print(f"\n=== {func_name.upper()} ===")
        for opt_name in ["eternal_optimizer", "adam_optimizer"]:
            opt_r = r[opt_name]
            print(f"  {opt_name}: final_value={opt_r['final_value']:.6e}, "
                  f"steps={opt_r['steps']}, converged={opt_r['converged']}")


if __name__ == "__main__":
    main()
