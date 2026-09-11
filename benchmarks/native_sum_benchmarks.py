"""Source-verified batch sum comparison with exact represented-input oracles.

Run from the repository root with PYTHONPATH=. and the native extension installed.
Input creation and exact oracles are outside timed regions unless named explicitly.
"""

import argparse
import gc
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from time import perf_counter_ns

import numpy as np

import balansis
from balansis import AbsoluteValue, Operations, sum_array
from balansis.array import native_available

ROOT = Path(__file__).resolve().parents[1]
NATIVE_SOURCES = [
    "pyproject.toml",
    "setup.py",
    "src/module.c",
    "src/neumaier.c",
    "src/neumaier.h",
]


def source_hash(base, paths):
    digest = hashlib.sha256()
    for name in paths:
        digest.update(name.encode() + b"\0" + (base / name).read_bytes())
    return digest.hexdigest()


def verify_sources():
    if not Path(balansis.__file__).resolve().is_relative_to(ROOT / "balansis"):
        raise RuntimeError("foreign Balansis package; run with PYTHONPATH=.")
    if not native_available():
        raise RuntimeError("native extension required; install ./native")
    import _balansis_kernels as kernels

    expected = source_hash(ROOT / "native", NATIVE_SOURCES)
    if kernels.SOURCE_SHA256 != expected:
        raise RuntimeError("stale native binary; rebuild and reinstall ./native")
    return kernels


def measure(methods):
    """Calibrate, shuffle method order per round, report medians and spread."""
    counts = {}
    samples = {name: [] for name in methods}
    for name, method in methods.items():
        method()
        count = 1
        while True:
            start = perf_counter_ns()
            for _ in range(count):
                method()
            elapsed = perf_counter_ns() - start
            if elapsed >= 5_000_000 or count >= 32768:
                break
            count *= 2
        counts[name] = count
    order_rng = random.Random(731)
    names = list(methods)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(9):
            order_rng.shuffle(names)
            for name in names:
                start = perf_counter_ns()
                for _ in range(counts[name]):
                    methods[name]()
                samples[name].append((perf_counter_ns() - start) / counts[name])
    finally:
        if gc_was_enabled:
            gc.enable()
    return {
        name: {
            "median_ns": statistics.median(values),
            "min_ns": min(values),
            "max_ns": max(values),
            "calls_per_round": counts[name],
            "rounds": len(values),
        }
        for name, values in samples.items()
    }


def evaluate(values, kernels):
    floats = values.tolist()
    objects = [AbsoluteValue.from_float(value) for value in floats]
    exact = sum(map(Fraction, floats), Fraction())
    rounded = float(exact)
    methods = {
        "legacy_prebuilt_objects": lambda: Operations.sequence_sum(objects)[
            0
        ].to_float(),
        "legacy_including_object_creation": lambda: Operations.sequence_sum(
            [AbsoluteValue.from_float(float(x)) for x in values]
        )[0].to_float(),
        "python_array_api": lambda: sum_array(values, backend="python")[0].to_float(),
        "native_array_api": lambda: sum_array(values, backend="native")[0].to_float(),
        "native_kernel": lambda: kernels.neumaier_sum(values)[0],
        "math_fsum_array": lambda: math.fsum(values),
        "math_fsum_prebuilt_list": lambda: math.fsum(floats),
        "numpy_sum": lambda: float(np.sum(values)),
    }
    timing = measure(methods)
    outcomes = {}
    for name, method in methods.items():
        result = method()
        outcomes[name] = {
            **timing[name],
            "result": result,
            "absolute_error_vs_exact": float(abs(Fraction(result) - exact)),
            "ulp_distance_from_rounded_reference": abs(result - rounded)
            / math.ulp(rounded),
            "matches_rounded_reference": result == rounded,
        }
    return {"size": len(values), "rounded_reference": rounded, "methods": outcomes}


def build_report():
    kernels = verify_sources()
    rng = np.random.default_rng(731)
    scenarios = []
    for size in [3, 1024, 100000]:
        ordinary = rng.normal(size=size)
        cancellation = np.zeros(size)
        cancellation[: size // 3 * 3] = np.tile([1e16, 1.0, -1e16], size // 3)
        for name, values in [("ordinary", ordinary), ("cancellation", cancellation)]:
            scenarios.append({"scenario": name, **evaluate(values, kernels)})
    python_files = sorted(
        path.relative_to(ROOT).as_posix() for path in (ROOT / "balansis").rglob("*.py")
    )
    compiler = subprocess.run(
        ["cc", "--version"], capture_output=True, text=True, check=False
    ).stdout.splitlines()
    cpu = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "artifact": "native_sum_comparison",
        "python": sys.version,
        "numpy": np.__version__,
        "system": platform.system(),
        "architecture": platform.machine(),
        "cpu": cpu,
        "system_cc": compiler[0] if compiler else "unknown",
        "kernel_source_sha256": kernels.SOURCE_SHA256,
        "kernel_binary_sha256": hashlib.sha256(
            Path(kernels.__file__).read_bytes()
        ).hexdigest(),
        "python_source_sha256": source_hash(ROOT, python_files),
        "methodology": {
            "oracle": "Fraction over each represented float64 input",
            "timing": "9 shuffled rounds; calibration >=5ms; GC disabled for all methods",
            "scope": "prepared inputs except explicitly named object creation",
            "limitations": "single environment; no SIMD; no cross-platform guarantee",
        },
        "scenarios": scenarios,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)
