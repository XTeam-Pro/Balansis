"""Claim artifacts must describe the checked-out implementation."""

import runpy
from pathlib import Path

import pytest

import balansis


def load_harness():
    root = Path(__file__).resolve().parents[1]
    return runpy.run_path(str(root / "benchmarks" / "claim_closure_benchmarks.py"))


def test_claim_report_cancels_exact_opposites_and_identifies_implementation():
    report = load_harness()["build_report"]()
    scenarios = {entry["scenario"]: entry for entry in report["scenarios"]}
    cancellation = scenarios["cancellation_signal"]
    assert cancellation["balansis_result"] == cancellation["expected_exact"] == 0.0
    assert cancellation["reversed_result"] == 0.0
    assert cancellation["result_is_absolute"]
    assert scenarios["large_scale_aggregation"]["balansis_result"] == 1.0
    assert len(bytes.fromhex(report["implementation_sha256"])) == 32
    assert report["python_version"]
    assert report["numpy_version"]


def test_claim_report_rejects_a_package_from_another_checkout(monkeypatch, tmp_path):
    harness = load_harness()
    monkeypatch.setattr(
        balansis, "__file__", str(tmp_path / "balansis" / "__init__.py")
    )
    with pytest.raises(RuntimeError, match="outside this checkout"):
        harness["build_report"]()
