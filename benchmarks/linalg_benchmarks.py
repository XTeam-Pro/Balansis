"""
Linear Algebra Benchmarks for Balansis ACT vs NumPy

Compares ACT-compensated linear algebra operations (GEMM, SVD, QR)
against NumPy implementations for ill-conditioned matrices.
"""

import json
import time
import math
from typing import Dict, Any, List, Optional

import numpy as np

from balansis.core.absolute import AbsoluteValue
from balansis.linalg.gemm import matmul
from balansis.linalg.svd import svd
from balansis.linalg.qr import qr_decompose


class LinalgBenchmark:
    """Benchmarks for ACT vs NumPy linear algebra operations."""

    SIZES = [10, 50, 100, 500]

    def __init__(self, sizes: Optional[List[int]] = None):
        self.sizes = sizes or self.SIZES

    @staticmethod
    def _make_ill_conditioned(n: int, cond: float = 1e10) -> np.ndarray:
        """Create an ill-conditioned matrix with specified condition number."""
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(n, n))
        V, _ = np.linalg.qr(np.random.randn(n, n))
        singular_values = np.logspace(0, -math.log10(cond), n)
        S = np.diag(singular_values)
        return U @ S @ V.T

    @staticmethod
    def _np_to_act(matrix: np.ndarray) -> List[List[AbsoluteValue]]:
        """Convert numpy matrix to ACT AbsoluteValue matrix."""
        return [
            [AbsoluteValue.from_float(float(matrix[i, j]))
             for j in range(matrix.shape[1])]
            for i in range(matrix.shape[0])
        ]

    @staticmethod
    def _act_to_np(matrix: List[List[AbsoluteValue]]) -> np.ndarray:
        """Convert ACT matrix to numpy array."""
        return np.array([
            [cell.to_float() for cell in row]
            for row in matrix
        ])

    def benchmark_gemm(self, n: int, cond: float = 1e10) -> Dict[str, Any]:
        """Benchmark GEMM: ACT vs numpy for ill-conditioned matrices."""
        A = self._make_ill_conditioned(n, cond)
        B = self._make_ill_conditioned(n, cond)

        # Reference: numpy float64
        t0 = time.perf_counter()
        C_np = A @ B
        numpy_time = time.perf_counter() - t0

        # ACT GEMM
        A_act = self._np_to_act(A)
        B_act = self._np_to_act(B)

        t0 = time.perf_counter()
        C_act_raw, comp_factor = matmul(A_act, B_act, use_compensation=True)
        act_time = time.perf_counter() - t0
        C_act = self._act_to_np(C_act_raw)

        # High-precision reference via float128 emulation
        A128 = A.astype(np.longdouble)
        B128 = B.astype(np.longdouble)
        C_ref = (A128 @ B128).astype(np.float64)

        numpy_err = float(np.linalg.norm(C_np - C_ref) / np.linalg.norm(C_ref))
        act_err = float(np.linalg.norm(C_act - C_ref) / np.linalg.norm(C_ref))

        return {
            "operation": "gemm",
            "size": n,
            "condition_number": cond,
            "numpy_reconstruction_error": numpy_err,
            "act_reconstruction_error": act_err,
            "stability_ratio": numpy_err / act_err if act_err > 0 else float("inf"),
            "numpy_time_s": numpy_time,
            "act_time_s": act_time,
        }

    def benchmark_svd(self, n: int, cond: float = 1e10) -> Dict[str, Any]:
        """Benchmark SVD reconstruction: ACT vs numpy."""
        A = self._make_ill_conditioned(n, cond)

        # NumPy SVD
        t0 = time.perf_counter()
        U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)
        numpy_time = time.perf_counter() - t0
        A_reconstructed_np = U_np @ np.diag(S_np) @ Vt_np
        numpy_err = float(np.linalg.norm(A - A_reconstructed_np) / np.linalg.norm(A))

        # ACT SVD
        A_act = self._np_to_act(A)
        t0 = time.perf_counter()
        U_act, S_act, Vt_act = svd(A_act)
        act_time = time.perf_counter() - t0

        U_np2 = self._act_to_np(U_act)
        S_np2 = np.array([s.to_float() for s in S_act])
        Vt_np2 = self._act_to_np(Vt_act)
        A_reconstructed_act = U_np2 @ np.diag(S_np2) @ Vt_np2
        act_err = float(np.linalg.norm(A - A_reconstructed_act) / np.linalg.norm(A))

        return {
            "operation": "svd",
            "size": n,
            "condition_number": cond,
            "numpy_reconstruction_error": numpy_err,
            "act_reconstruction_error": act_err,
            "stability_ratio": numpy_err / act_err if act_err > 0 else float("inf"),
            "numpy_time_s": numpy_time,
            "act_time_s": act_time,
        }

    def benchmark_qr(self, n: int, cond: float = 1e10) -> Dict[str, Any]:
        """Benchmark QR orthogonality: ACT vs numpy."""
        A = self._make_ill_conditioned(n, cond)

        # NumPy QR
        t0 = time.perf_counter()
        Q_np, R_np = np.linalg.qr(A)
        numpy_time = time.perf_counter() - t0
        I_n = np.eye(n)
        numpy_orth_err = float(np.linalg.norm(Q_np.T @ Q_np - I_n))

        # ACT QR
        A_act = self._np_to_act(A)
        t0 = time.perf_counter()
        Q_act, R_act = qr_decompose(A_act)
        act_time = time.perf_counter() - t0

        Q_np2 = self._act_to_np(Q_act)
        act_orth_err = float(np.linalg.norm(Q_np2.T @ Q_np2 - I_n))

        return {
            "operation": "qr",
            "size": n,
            "condition_number": cond,
            "numpy_orthogonality_error": numpy_orth_err,
            "act_orthogonality_error": act_orth_err,
            "stability_ratio": numpy_orth_err / act_orth_err if act_orth_err > 0 else float("inf"),
            "numpy_time_s": numpy_time,
            "act_time_s": act_time,
        }

    def run_all(self) -> Dict[str, Any]:
        """Run the full linalg benchmark suite."""
        results: Dict[str, Any] = {
            "gemm": [],
            "svd": [],
            "qr": [],
        }

        for n in self.sizes:
            print(f"  LinAlg benchmarks for size {n}x{n}...")
            results["gemm"].append(self.benchmark_gemm(n))
            results["svd"].append(self.benchmark_svd(n))
            results["qr"].append(self.benchmark_qr(n))

        return results

    def to_json(self) -> str:
        """Run benchmarks and return JSON string."""
        results = self.run_all()
        return json.dumps(results, indent=2, default=str)


def main():
    print("Running linear algebra benchmarks...")
    bench = LinalgBenchmark()
    results = bench.run_all()

    with open("linalg_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nResults saved to linalg_benchmark_results.json")

    for op, op_results in results.items():
        print(f"\n=== {op.upper()} ===")
        for r in op_results:
            size = r["size"]
            ratio = r.get("stability_ratio", "N/A")
            print(f"  {size}x{size}: stability_ratio = {ratio:.2f}" if isinstance(ratio, float) else f"  {size}x{size}: stability_ratio = {ratio}")


if __name__ == "__main__":
    main()
