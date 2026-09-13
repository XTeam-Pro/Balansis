import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.sep + "..")
try:
    import numpy as np
except ImportError:
    np = None
from balansis.core.absolute import AbsoluteValue
from balansis.linalg.gemm import matmul


def bench_add_float(n=100000):
    s = 0.0
    for i in range(n):
        s += 1.0
    return s


def bench_add_absolute(n=100000):
    s = AbsoluteValue.absolute()
    one = AbsoluteValue.from_float(1.0)
    for i in range(n):
        s = s + one
    return s


def bench_numpy_add(n=100000):
    if np is None:
        return None
    a = np.ones(n, dtype=float)
    return float(np.sum(a))


def bench_gemm_absolute():
    a = [
        [AbsoluteValue.from_float(1.0), AbsoluteValue.from_float(2.0)],
        [AbsoluteValue.from_float(3.0), AbsoluteValue.from_float(4.0)],
    ]
    b = [
        [AbsoluteValue.from_float(5.0), AbsoluteValue.from_float(6.0)],
        [AbsoluteValue.from_float(7.0), AbsoluteValue.from_float(8.0)],
    ]
    return matmul(a, b)


def main():
    for name, fn in [
        ("add_float", bench_add_float),
        ("add_absolute", bench_add_absolute),
        ("numpy_add", bench_numpy_add),
    ]:
        t0 = time.time()
        r = fn()
        t1 = time.time()
        if r is None:
            print(name, "skipped (numpy not installed)")
        else:
            print(
                name,
                r if name != "add_absolute" else r.to_float(),
                round((t1 - t0) * 1000, 2),
                "ms",
            )
    t0 = time.time()
    c = bench_gemm_absolute()
    t1 = time.time()
    print(
        "gemm_absolute",
        [[x.to_float() for x in row] for row in c],
        round((t1 - t0) * 1000, 2),
        "ms",
    )


if __name__ == "__main__":
    main()
