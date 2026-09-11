"""A native performance claim must identify the source actually executed."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_foreign_python_source_is_rejected(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "benchmarks/native_sum_benchmarks.py"
    harness = runpy.run_path(str(path))
    monkeypatch.setattr(
        harness["balansis"], "__file__", "/foreign/balansis/__init__.py"
    )
    with pytest.raises(RuntimeError, match="foreign Balansis"):
        harness["verify_sources"]()


def test_stale_native_binary_is_rejected(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "benchmarks/native_sum_benchmarks.py"
    harness = runpy.run_path(str(path))
    verify = harness["verify_sources"]
    monkeypatch.setitem(verify.__globals__, "native_available", lambda: True)
    monkeypatch.setitem(
        harness["sys"].modules,
        "_balansis_kernels",
        SimpleNamespace(SOURCE_SHA256="stale"),
    )
    with pytest.raises(RuntimeError, match="stale native binary"):
        verify()
