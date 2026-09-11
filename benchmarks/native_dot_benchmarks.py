"""Exact dot-product and Jacobi SVD comparison, synthetic inputs only."""

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import runpy
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from balansis import dot_array
from balansis.core._eft import dot2, two_product_arr

ROOT = Path(__file__).resolve().parents[1]
# Load directly: the unrelated legacy benchmarks package initializer imports
# optional benchmark modules that are not shipped in every checkout.
SHARED = runpy.run_path(str(ROOT / "benchmarks/native_sum_benchmarks.py"))


def legacy_dot(a, b):
    """Pre-native dot2 algorithm, retained here solely as the benchmark baseline."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0:
        return 0.0
    p, e = two_product_arr(a, b)
    finite = np.isfinite(p) & np.isfinite(e)
    if not finite.all():
        return float(np.dot(a, b))
    return math.fsum(np.concatenate([p, e]))


def reference(a, b):
    return sum(
        (Fraction(float(x)) * Fraction(float(y)) for x, y in zip(a, b)), Fraction()
    )


def outcome(function, exact):
    try:
        with np.errstate(all="ignore"):
            result = function()
    except (ValueError, OverflowError) as exc:
        return {"exception": type(exc).__name__}
    if not math.isfinite(result):
        return {"result": str(result), "matches_rounded_reference": False}
    return {
        "result": result,
        "matches_rounded_reference": result == float(exact),
        "absolute_error_vs_exact": float(abs(Fraction(result) - exact)),
    }


def svd_comparison(matrix):
    module = importlib.import_module("balansis.linalg.svd")
    original = module.dot2

    def run(kernel):
        module.dot2 = kernel
        try:
            return module._act_jacobi_svd(matrix)
        finally:
            module.dot2 = original

    methods = {
        "legacy_dot_svd": lambda: run(legacy_dot),
        "native_dot_svd": lambda: run(dot2),
    }
    timing = SHARED["measure"](methods)
    result = {}
    for name, function in methods.items():
        u, s, vt = function()
        result[name] = {
            **timing[name],
            "relative_reconstruction_error": float(
                np.linalg.norm(u @ np.diag(s) @ vt - matrix) / np.linalg.norm(matrix)
            ),
            "u_orthogonality_error": float(
                np.linalg.norm(u.T @ u - np.eye(u.shape[1]))
            ),
            "singular_values": s.tolist(),
        }
    return {"shape": list(matrix.shape), "methods": result}


def build_report():
    kernels = SHARED["verify_sources"]()
    if not hasattr(kernels, "exact_dot"):
        raise RuntimeError("exact dot kernel required; rebuild ./native")
    rng = np.random.default_rng(840)
    samples = []
    for n in [3, 1024, 100000]:
        ordinary = (rng.normal(size=n), rng.normal(size=n))
        cancellation = np.zeros(n)
        cancellation[: n // 3 * 3] = np.tile([1e16, 1, -1e16], n // 3)
        for name, (a, b) in [
            ("ordinary", ordinary),
            ("cancellation", (cancellation, np.ones(n))),
        ]:
            exact = reference(a, b)
            methods = {
                "legacy_dot2": lambda: legacy_dot(a, b),
                "native_array_api": lambda: dot_array(a, b, backend="native"),
                "native_kernel": lambda: kernels.exact_dot(a, b),
                "integrated_dot2": lambda: dot2(a, b),
                "python_exact_api": lambda: dot_array(a, b, backend="python"),
                "numpy_dot": lambda: float(np.dot(a, b)),
                "fsum_rounded_products": lambda: math.fsum(a * b),
            }
            timing = SHARED["measure"](methods)
            samples.append(
                {
                    "scenario": name,
                    "size": n,
                    "rounded_reference": float(exact),
                    "methods": {
                        key: {**timing[key], **outcome(fn, exact)}
                        for key, fn in methods.items()
                    },
                }
            )
    tiny = math.ulp(0.0)
    boundaries = []
    for name, a, b in [
        ("subnormal_products", [tiny, tiny], [0.5, 0.5]),
        ("overflowing_products_cancel", [1e308, 1e308], [2, -2]),
        (
            "product_rounding_residual",
            [math.nextafter(1, 2), -1],
            [math.nextafter(1, 0), 1],
        ),
    ]:
        exact = reference(a, b)
        boundaries.append(
            {
                "scenario": name,
                "rounded_reference": float(exact),
                "legacy": outcome(lambda: legacy_dot(a, b), exact),
                "native": outcome(lambda: dot_array(a, b, backend="native"), exact),
            }
        )
    python_paths = sorted(
        p.relative_to(ROOT).as_posix() for p in (ROOT / "balansis").rglob("*.py")
    )
    svd = [
        svd_comparison(rng.normal(size=shape)) for shape in [(16, 4), (64, 8), (256, 8)]
    ]
    cpu = "unknown"
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "artifact": "native_exact_dot_comparison",
        "python": sys.version,
        "numpy": np.__version__,
        "architecture": platform.machine(),
        "cpu": cpu,
        "blas_thread_environment": {
            name: os.environ.get(name, "unset")
            for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]
        },
        "kernel_source_sha256": kernels.SOURCE_SHA256,
        "kernel_binary_sha256": hashlib.sha256(
            Path(kernels.__file__).read_bytes()
        ).hexdigest(),
        "python_source_sha256": SHARED["source_hash"](ROOT, python_paths),
        "methodology": "seed 840; Fraction oracle; prepared inputs; nine shuffled calibrated rounds; GC disabled; shared host, no CPU pinning; SVD baseline substitutes only dot2",
        "scenarios": samples,
        "boundaries": boundaries,
        "svd": svd,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)
