"""Reproducible benchmark and scenario harness for documented Balansis claims.

This script focuses on the public scenarios currently surfaced in README and
guides. It produces both behavioral outputs and lightweight timing data so that
claims about stability and engineering tradeoffs are backed by runnable assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path

import balansis
import numpy as np

from balansis import AbsoluteValue, Operations
from balansis.core.eternity import SingularPolicy
from balansis.logic.compensator import Compensator
from balansis.finance.ledger import Ledger
from balansis.linalg.svd import svd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmarks" / "results" / "claim_closure_baseline.json"


@dataclass
class TimingSummary:
    mean_seconds: float
    median_seconds: float
    min_seconds: float
    max_seconds: float
    iterations: int


def naive_float_sum(values: list[float]) -> float:
    total = 0.0
    for value in values:
        total += value
    return total


def measure(fn, iterations: int = 200) -> TimingSummary:
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return TimingSummary(
        mean_seconds=statistics.mean(samples),
        median_seconds=statistics.median(samples),
        min_seconds=min(samples),
        max_seconds=max(samples),
        iterations=iterations,
    )


def scenario_large_scale_aggregation() -> dict[str, object]:
    float_values = [1e16, 1.0, -1e16]
    balansis_values = [AbsoluteValue.from_float(v) for v in float_values]

    builtin_sum_result = sum(float_values)
    naive_sum_result = naive_float_sum(float_values)
    act_result, compensation = Operations.sequence_sum(balansis_values)

    return {
        "scenario": "large_scale_aggregation",
        "builtin_sum_result": builtin_sum_result,
        "naive_float_result": naive_sum_result,
        "balansis_result": act_result.to_float(),
        "balansis_compensation": compensation,
        "expected_exact": 1.0,
        "residual_preserved_vs_naive_float": act_result.to_float() == 1.0
        and naive_sum_result == 0.0,
        "timing": {
            "builtin_sum": asdict(measure(lambda: sum(float_values))),
            "naive_float_sum": asdict(measure(lambda: naive_float_sum(float_values))),
            "balansis_sequence_sum": asdict(
                measure(lambda: Operations.sequence_sum(balansis_values))
            ),
        },
    }


def scenario_cancellation_signal() -> dict[str, object]:
    float_result = 1e16 + (-1e16)
    left = AbsoluteValue.from_float(1e16)
    right = AbsoluteValue.from_float(-1e16)
    act_result, compensation = Operations.compensated_add(left, right)
    reversed_result, _ = Operations.compensated_add(right, left)

    return {
        "scenario": "cancellation_signal",
        "float_result": float_result,
        "balansis_result": act_result.to_float(),
        "balansis_magnitude": act_result.magnitude,
        "balansis_direction": act_result.direction,
        "balansis_compensation": compensation,
        "result_is_absolute": act_result.is_absolute(),
        "expected_exact": 0.0,
        "reversed_result": reversed_result.to_float(),
        "rounded_before_construction": (1e16 + 1.0) - 1e16,
        "timing": {
            "float_expr": asdict(measure(lambda: 1e16 + (-1e16))),
            "balansis_compensated_add": asdict(
                measure(lambda: Operations.compensated_add(left, right))
            ),
        },
    }


def scenario_finance_zero_sum() -> dict[str, object]:
    def build_ledger() -> Ledger:
        ledger = Ledger()
        ledger.post_entry("cash", Decimal("250.00"))
        ledger.post_entry("cash", Decimal("-250.00"))
        return ledger

    ledger = build_ledger()
    balance = ledger.balance()

    return {
        "scenario": "finance_zero_sum",
        "balance_is_absolute": balance.is_absolute(),
        "balance_float": balance.to_float(),
        "timing": {
            "ledger_balance": asdict(measure(lambda: build_ledger().balance())),
        },
    }


def scenario_division_contract() -> dict[str, object]:
    numerator = AbsoluteValue.from_float(6.0)
    denominator = AbsoluteValue.from_float(2.0)
    ratio, compensation = Operations.compensated_divide(numerator, denominator)

    absolute_denominator_error = None
    try:
        Operations.compensated_divide(numerator, AbsoluteValue.absolute())
    except ValueError as exc:
        absolute_denominator_error = str(exc)

    return {
        "scenario": "division_contract",
        "ratio_value": ratio.numerical_value(),
        "ratio_signed_value": ratio.signed_value(),
        "compensation": compensation,
        "absolute_denominator_error": absolute_denominator_error,
        "timing": {
            "valid_division": asdict(
                measure(lambda: Operations.compensated_divide(numerator, denominator))
            ),
        },
    }


def scenario_extended_division_states() -> dict[str, object]:
    finite_ratio, finite_compensation = Operations.compensated_divide_extended(
        AbsoluteValue.from_float(6.0),
        AbsoluteValue.from_float(2.0),
    )
    infinite_ratio, infinite_compensation = Operations.compensated_divide_extended(
        AbsoluteValue.from_float(6.0),
        AbsoluteValue.absolute(),
    )
    indeterminate_ratio, indeterminate_compensation = Operations.compensated_divide_extended(
        AbsoluteValue.absolute(),
        AbsoluteValue.absolute(),
    )

    return {
        "scenario": "extended_division_states",
        "finite_kind": finite_ratio.kind,
        "finite_value": finite_ratio.numerical_value(),
        "finite_compensation": finite_compensation,
        "infinite_kind": infinite_ratio.kind,
        "infinite_direction": infinite_ratio.direction,
        "infinite_compensation": infinite_compensation,
        "indeterminate_kind": indeterminate_ratio.kind,
        "indeterminate_reason": indeterminate_ratio.reason,
        "indeterminate_compensation": indeterminate_compensation,
        "timing": {
            "finite_division_extended": asdict(
                measure(
                    lambda: Operations.compensated_divide_extended(
                        AbsoluteValue.from_float(6.0),
                        AbsoluteValue.from_float(2.0),
                    )
                )
            ),
            "infinite_division_extended": asdict(
                measure(
                    lambda: Operations.compensated_divide_extended(
                        AbsoluteValue.from_float(6.0),
                        AbsoluteValue.absolute(),
                    )
                )
            ),
        },
    }


def scenario_policy_driven_singular_arithmetic() -> dict[str, object]:
    numerator = AbsoluteValue.from_float(8.0)
    denominator = AbsoluteValue.absolute()

    propagated_ratio, propagated_compensation, propagated_event = Operations.compensated_divide_policy(
        numerator,
        denominator,
        SingularPolicy.PROPAGATE,
    )
    saturated_ratio, saturated_compensation, saturated_event = Operations.compensated_divide_policy(
        numerator,
        denominator,
        SingularPolicy.SATURATE,
        saturation_limit=100.0,
    )

    compensator = Compensator()
    _, telemetry_event = compensator.compensate_division_policy(
        numerator,
        denominator,
        SingularPolicy.SATURATE,
        saturation_limit=100.0,
    )
    telemetry = compensator.get_singular_telemetry()

    return {
        "scenario": "policy_driven_singular_arithmetic",
        "propagate_kind": propagated_ratio.kind,
        "propagate_event_policy": None if propagated_event is None else propagated_event.policy.value,
        "propagate_compensation": propagated_compensation,
        "saturate_kind": saturated_ratio.kind,
        "saturate_value": saturated_ratio.numerical_value(),
        "saturate_event_policy": None if saturated_event is None else saturated_event.policy.value,
        "saturate_compensation": saturated_compensation,
        "telemetry_event_policy": None if telemetry_event is None else telemetry_event.policy.value,
        "telemetry_singular_operations": telemetry["singular_operations"],
        "telemetry_policy_event_count": len(telemetry["policy_events"]),
        "timing": {
            "propagate_policy_division": asdict(
                measure(
                    lambda: Operations.compensated_divide_policy(
                        numerator,
                        denominator,
                        SingularPolicy.PROPAGATE,
                    )
                )
            ),
            "saturate_policy_division": asdict(
                measure(
                    lambda: Operations.compensated_divide_policy(
                        numerator,
                        denominator,
                        SingularPolicy.SATURATE,
                        saturation_limit=100.0,
                    )
                )
            ),
        },
    }


def scenario_pipeline_policy_propagation() -> dict[str, object]:
    matrix = [
        [AbsoluteValue.absolute(), AbsoluteValue.absolute()],
        [AbsoluteValue.absolute(), AbsoluteValue.absolute()],
    ]
    propagated = svd(matrix, singular_policy=SingularPolicy.PROPAGATE)
    saturated = svd(matrix, singular_policy=SingularPolicy.SATURATE, saturation_limit=50.0)

    propagated_telemetry = propagated.singular_telemetry()
    saturated_telemetry = saturated.singular_telemetry()

    return {
        "scenario": "pipeline_policy_propagation",
        "svd_propagate_event_count": len(propagated_telemetry),
        "svd_propagate_first_policy": propagated_telemetry[0]["policy"] if propagated_telemetry else None,
        "svd_saturate_event_count": len(saturated_telemetry),
        "svd_saturate_first_policy": saturated_telemetry[0]["policy"] if saturated_telemetry else None,
        "svd_reconstruction_error": propagated.reconstruction_error,
        "timing": {
            "svd_policy_propagate": asdict(
                measure(lambda: svd(matrix, singular_policy=SingularPolicy.PROPAGATE))
            ),
            "svd_policy_saturate": asdict(
                measure(lambda: svd(matrix, singular_policy=SingularPolicy.SATURATE, saturation_limit=50.0))
            ),
        },
    }


def build_report() -> dict[str, object]:
    if not Path(balansis.__file__).resolve().is_relative_to(ROOT / "balansis"):
        raise RuntimeError(
            "Benchmark imported Balansis outside this checkout. Run from the "
            "repository root with PYTHONPATH=. python benchmarks/claim_closure_benchmarks.py"
        )
    source_hash = hashlib.sha256()
    for source in sorted((ROOT / "balansis").rglob("*.py")):
        source_hash.update(source.relative_to(ROOT).as_posix().encode() + b"\0")
        source_hash.update(source.read_bytes() + b"\0")
    scenarios = [
        scenario_large_scale_aggregation(),
        scenario_cancellation_signal(),
        scenario_finance_zero_sum(),
        scenario_division_contract(),
        scenario_extended_division_states(),
        scenario_policy_driven_singular_arithmetic(),
        scenario_pipeline_policy_propagation(),
    ]
    return {
        "artifact": "claim_closure_baseline",
        "version": "1.1.0",
        "implementation_sha256": source_hash.hexdigest(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run documented Balansis claim scenarios.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the JSON artifact.",
    )
    args = parser.parse_args()

    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote benchmark artifact to {args.output}")


if __name__ == "__main__":
    main()
