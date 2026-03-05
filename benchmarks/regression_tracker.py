"""
Regression Tracker for Balansis Benchmarks

Compares current benchmark results against a stored baseline.
Fails (exit code 1) if regression exceeds threshold.
Updates baseline with --update flag.

Usage:
    python -m benchmarks.regression_tracker [--update]
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, Optional

from benchmarks.accuracy_benchmarks import AccuracyBenchmark
from benchmarks.linalg_benchmarks import LinalgBenchmark

BASELINES_PATH = os.path.join(os.path.dirname(__file__), "baselines.json")
REGRESSION_THRESHOLD = 0.05  # 5%


def load_baselines(path: str = BASELINES_PATH) -> Dict[str, Any]:
    """Load baseline values from JSON file."""
    if not os.path.exists(path):
        print(f"WARNING: Baselines file not found at {path}")
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_baselines(data: Dict[str, Any], path: str = BASELINES_PATH) -> None:
    """Save baseline values to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Baselines saved to {path}")


def run_accuracy_benchmarks() -> Dict[str, float]:
    """Run accuracy benchmarks and extract key metrics."""
    bench = AccuracyBenchmark()

    metrics: Dict[str, float] = {}

    # Catastrophic cancellation scenario
    result = bench.run_accuracy_test(1000, "catastrophic_cancellation")
    act_err = result["methods"]["act"]["absolute_error"]
    f64_err = result["methods"]["float64"]["absolute_error"]
    metrics["catastrophic_cancellation_stability_ratio"] = (
        f64_err / act_err if act_err > 0 else float("inf")
    )

    # Catastrophic near-zero scenario
    result_nz = bench.run_accuracy_test(1000, "catastrophic_near_zero")
    act_err_nz = result_nz["methods"]["act"]["absolute_error"]
    f64_err_nz = result_nz["methods"]["float64"]["absolute_error"]
    metrics["catastrophic_near_zero_stability_ratio"] = (
        f64_err_nz / act_err_nz if act_err_nz > 0 else float("inf")
    )

    # Ill-conditioned scenario
    result_ic = bench.run_accuracy_test(100, "ill_conditioned_matrix")
    act_err_ic = result_ic["methods"]["act"]["absolute_error"]
    f64_err_ic = result_ic["methods"]["float64"]["absolute_error"]
    metrics["ill_conditioned_stability_ratio"] = (
        f64_err_ic / act_err_ic if act_err_ic > 0 else float("inf")
    )

    return metrics


def run_linalg_benchmarks() -> Dict[str, float]:
    """Run linalg benchmarks and extract key metrics."""
    bench = LinalgBenchmark(sizes=[10, 50])
    results = bench.run_all()

    metrics: Dict[str, float] = {}

    # GEMM reconstruction error (smallest size for speed)
    if results["gemm"]:
        metrics["gemm_reconstruction_error_act"] = results["gemm"][0]["act_reconstruction_error"]

    # SVD reconstruction error
    if results["svd"]:
        metrics["svd_reconstruction_error_act"] = results["svd"][0]["act_reconstruction_error"]

    # QR orthogonality error
    if results["qr"]:
        metrics["qr_orthogonality_error_act"] = results["qr"][0]["act_orthogonality_error"]

    return metrics


def compare_metrics(
    current: Dict[str, float],
    baseline: Dict[str, float],
    threshold: float = REGRESSION_THRESHOLD,
) -> list:
    """Compare current metrics against baseline.

    Returns list of regression descriptions (empty = no regressions).
    For stability_ratio metrics, higher is better (regression = decrease).
    For error metrics, lower is better (regression = increase).
    """
    regressions = []

    for key, current_val in current.items():
        if key not in baseline:
            continue

        baseline_val = baseline[key]

        if baseline_val == 0 or not all(
            map(lambda x: x != float("inf") and x != float("-inf"),
                [current_val, baseline_val])
        ):
            continue

        if "stability_ratio" in key:
            # Higher is better — regression if current is lower
            if current_val < baseline_val * (1 - threshold):
                pct = (1 - current_val / baseline_val) * 100
                regressions.append(
                    f"REGRESSION: {key}: {baseline_val:.4g} -> {current_val:.4g} "
                    f"({pct:.1f}% decrease)"
                )
        else:
            # Lower is better — regression if current is higher
            if current_val > baseline_val * (1 + threshold):
                pct = (current_val / baseline_val - 1) * 100
                regressions.append(
                    f"REGRESSION: {key}: {baseline_val:.4g} -> {current_val:.4g} "
                    f"({pct:.1f}% increase)"
                )

    return regressions


class RegressionTracker:
    """Track benchmark regressions against stored baselines."""

    def __init__(self, baselines_path: str = BASELINES_PATH):
        self.baselines_path = baselines_path

    def run(self, update: bool = False) -> int:
        """Run benchmarks, compare with baselines, return exit code."""
        print("=" * 60)
        print("Balansis Regression Tracker")
        print("=" * 60)

        baselines = load_baselines(self.baselines_path)

        print("\nRunning accuracy benchmarks...")
        accuracy_metrics = run_accuracy_benchmarks()
        print("Running linalg benchmarks...")
        linalg_metrics = run_linalg_benchmarks()

        current = {
            "accuracy": accuracy_metrics,
            "linalg": linalg_metrics,
        }

        print("\n--- Current Metrics ---")
        for category, metrics in current.items():
            print(f"\n  [{category}]")
            for k, v in metrics.items():
                print(f"    {k}: {v:.6g}")

        if update:
            new_baselines = {
                "version": "0.5.0",
                "timestamp": time.strftime("%Y-%m-%d"),
                "accuracy": accuracy_metrics,
                "linalg": linalg_metrics,
            }
            save_baselines(new_baselines, self.baselines_path)
            print("\nBaselines updated successfully.")
            return 0

        if not baselines:
            print("\nNo baselines found. Run with --update to create initial baselines.")
            return 0

        # Compare
        all_regressions = []
        for category in ["accuracy", "linalg"]:
            if category in baselines and category in current:
                regressions = compare_metrics(
                    current[category], baselines[category]
                )
                all_regressions.extend(regressions)

        # Check stability ratio threshold
        stability_failures = []
        for key, val in accuracy_metrics.items():
            if "stability_ratio" in key and val < 10.0:
                stability_failures.append(
                    f"STABILITY FAILURE: {key} = {val:.2f} (required >= 10.0)"
                )

        print("\n--- Comparison Results ---")
        if not all_regressions and not stability_failures:
            print("  All metrics within acceptable range.")
            return 0

        for r in all_regressions:
            print(f"  {r}")
        for s in stability_failures:
            print(f"  {s}")

        total_failures = len(all_regressions) + len(stability_failures)
        print(f"\n  {total_failures} failure(s) detected.")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Balansis Regression Tracker")
    parser.add_argument(
        "--update", action="store_true",
        help="Update baselines with current benchmark results"
    )
    args = parser.parse_args()

    tracker = RegressionTracker()
    sys.exit(tracker.run(update=args.update))


if __name__ == "__main__":
    main()
