"""Fused exact Gram pair vs three exact calls, including complete Jacobi SVD."""

import argparse
import hashlib
import importlib
import json
import os
import platform
import runpy
import sys
from pathlib import Path

import numpy as np

import balansis.array as batch
from balansis.core._eft import dot2

ROOT = Path(__file__).resolve().parents[1]
SHARED = runpy.run_path(str(ROOT / "benchmarks/native_sum_benchmarks.py"))
REFERENCE = runpy.run_path(str(ROOT / "benchmarks/_svd_before_gram.py"))[
    "_act_jacobi_svd"
]


def three_dots(a, b):
    return dot2(a, a), dot2(b, b), dot2(a, b)


def build_report():
    kernels = SHARED["verify_sources"]()
    if not batch.native_gram_available():
        raise RuntimeError("fused Gram kernel required; rebuild ./native")
    rng = np.random.default_rng(774)
    vectors = []
    for size in [3, 1024, 100000]:
        a, b = rng.normal(size=size), rng.normal(size=size)
        methods = {
            "three_integrated_dots": lambda: three_dots(a, b),
            "three_native_kernels": lambda: (
                kernels.exact_dot(a, a),
                kernels.exact_dot(b, b),
                kernels.exact_dot(a, b),
            ),
            "fused_array_api": lambda: batch.gram_pair(a, b, backend="native"),
            "fused_native_kernel": lambda: kernels.gram_pair(a, b),
        }
        expected = three_dots(a, b)
        for function in methods.values():
            assert function() == expected
        vectors.append(
            {"size": size, "result": expected, "methods": SHARED["measure"](methods)}
        )
    module = importlib.import_module("balansis.linalg.svd")
    original_gram = batch.gram_pair
    svd = []
    for shape in [(16, 4), (64, 8), (256, 8), (8, 64), (1024, 8)]:
        matrix = rng.normal(size=shape)

        def layout_only():
            batch.gram_pair = three_dots
            try:
                return module._act_jacobi_svd(matrix)
            finally:
                batch.gram_pair = original_gram

        methods = {
            "previous_native_svd": lambda: REFERENCE(matrix),
            "column_layout_three_dots": layout_only,
            "fused_native_svd": lambda: module._act_jacobi_svd(matrix),
        }
        expected = REFERENCE(matrix)
        metrics = {}
        for name, function in methods.items():
            actual = function()
            equal = all(
                x.shape == y.shape and x.tobytes() == y.tobytes()
                for x, y in zip(expected, actual)
            )
            assert equal, (shape, name)
            u, s, vt = actual
            metrics[name] = {
                "bitwise_equal_to_previous": equal,
                "relative_reconstruction_error": float(
                    np.linalg.norm(u @ np.diag(s) @ vt - matrix)
                    / np.linalg.norm(matrix)
                ),
                "u_orthogonality_error": float(
                    np.linalg.norm(u.T @ u - np.eye(u.shape[1]))
                ),
            }
        timings = SHARED["measure"](methods)
        svd.append(
            {
                "shape": shape,
                "methods": {
                    name: {**timings[name], **metrics[name]} for name in methods
                },
            }
        )
    cpu = "unknown"
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    python_paths = sorted(
        p.relative_to(ROOT).as_posix() for p in (ROOT / "balansis").rglob("*.py")
    )
    return {
        "artifact": "native_gram_comparison",
        "python": sys.version,
        "numpy": np.__version__,
        "architecture": platform.machine(),
        "cpu": cpu,
        "threads": {
            k: os.environ.get(k, "unset")
            for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]
        },
        "kernel_source_sha256": kernels.SOURCE_SHA256,
        "kernel_binary_sha256": hashlib.sha256(
            Path(kernels.__file__).read_bytes()
        ).hexdigest(),
        "python_source_sha256": SHARED["source_hash"](ROOT, python_paths),
        "baseline_source_sha256": hashlib.sha256(
            (ROOT / "benchmarks/_svd_before_gram.py").read_bytes()
        ).hexdigest(),
        "methodology": "seed 774; prepared vectors; nine shuffled calibrated rounds; GC disabled; shared host without CPU pinning; previous SVD uses current exact C dots; layout-only ablation retains three dots",
        "vectors": vectors,
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
