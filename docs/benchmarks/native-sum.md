# Native batch summation

`balansis.sum_array` provides sequential Neumaier summation over a real 1-D
array. The optional C extension removes Python iteration and per-element object
access. It preserves operation order and diagnostic scaling; it does not change
`Operations.sequence_sum` or silently accelerate its existing object callers.

## Using it

From the source checkout, install the optional extension in the same environment
as Balansis:

```bash
python -m pip install ./native
python -c "from balansis.array import native_available; assert native_available()"
```

```python
import numpy as np
from balansis import sum_array

values = np.array([1e16, 1.0, -1e16], dtype=np.float64)
result, diagnostic = sum_array(values, backend="native")
assert result.to_float() == 1.0
```

Use `backend="python"` for the reference implementation and `backend="auto"`
(default) for native dispatch with a Python fallback. A missing or incompatible
native API falls back in auto mode. A broken native import or numerical exception
is surfaced, not hidden by a second implementation.

The public result is `(AbsoluteValue, diagnostic)`. Diagnostic means
`abs(accumulated_correction) / Operations.COMPENSATION_THRESHOLD`, exactly as in
the existing compensated sequence API; it is not an error bound and can itself
overflow for very large corrections. The kernel's private second output is the
signed correction, not this scaled diagnostic.

## Numerical and buffer contract

- Inputs must be one-dimensional real numeric arrays. Other ranks and complex,
  object or string dtypes are rejected. Integer and lower/higher-precision inputs
  are converted to binary64; this can lose information before summation.
- A contiguous native float64 ndarray needs no input copy. Strided or non-native
  arrays are converted. The C buffer interface also accepts unaligned and read-only
  double buffers, without writing to them.
- Neumaier compensation is not arbitrary precision and does not promise correct
  rounding on every input. Round-to-nearest-even and gradual underflow are assumed.
- Nonfinite inputs raise `ValueError`. Overflow of an intermediate sum, correction
  or final result raises `OverflowError`, even when a later term could cancel it.
  Input is inspected in order; the first detected invalid operation determines the
  exception. This is the new array API's explicit domain, not a change to legacy
  error handling.
- The kernel keeps the GIL. External/native writers must not mutate the buffer
  during a call. Parallel scheduling, SIMD and handwritten assembly are outside
  this implementation.

GCC/Clang builds explicitly disable fast-math and floating-point contraction;
the C source rejects known unsafe fast-math and excess-precision configurations.
This follows the requirements of compensated arithmetic rather than enabling
algebraic reassociation for throughput. See [GCC optimization options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)
and [CPython's buffer contract](https://docs.python.org/3/c-api/buffer.html).

## Reproducing the comparison

```bash
PYTHONPATH=. python benchmarks/native_sum_benchmarks.py \
  --output benchmarks/results/native_sum_comparison.json
```

The harness rejects imports from another Balansis checkout and native binaries
whose embedded source hash does not match the checkout. It records source and
binary hashes, Python, NumPy, CPU and the system compiler. Synthetic inputs use
seed 731. The oracle sums each represented binary64 input exactly with `Fraction`;
this work is outside the timed region.

Each method is warmed, calibrated, and measured in nine rounds with shuffled
method order. The report contains medians and min/max, with cyclic GC disabled
uniformly during measurement. Prepared lists, object sequences and NumPy arrays
are explicitly distinguished. Only `legacy_including_object_creation` includes
constructing `AbsoluteValue` objects; all API timings include return handling.

The [recorded results](../../benchmarks/results/native_sum_comparison.json) cover
ordinary normally distributed values and repeated `[1e16, 1, -1e16]` triplets,
with any trailing slots set to zero. Timings are observations from one environment,
not portable speed guarantees. CPU frequency and other host workloads are not
controlled. Do not compare a bare kernel to a full API without naming the boundary.

## Validation

```bash
python -m pytest tests/
python -m pytest tests/test_array_sum.py tests/test_native_benchmark_provenance.py --no-cov
```

The tests cover independent exact oracles, parity with the existing sequence
algorithm on seeded wide-exponent inputs, subnormals, exact cancellation,
intermediate/final overflow, malformed buffer types, read-only and unaligned
buffers, conversion, zero-copy forwarding and explicit backend selection. Also
run the focused tests in an environment without the extension to exercise real
fallback imports.

A standalone C harness can be checked with AddressSanitizer and UBSan:

```bash
cc -std=c11 -O2 -fno-fast-math -ffp-contract=off \
  -fexcess-precision=standard -fsanitize=address,undefined \
  -fno-omit-frame-pointer -I native/src \
  native/src/neumaier.c native/tests/test_neumaier.c -o /tmp/balansis-sum-check
/tmp/balansis-sum-check
```

Build both the Python package and the separate native distribution, then check
the installed wheels from outside the source checkout. Native compatibility on
other compilers, operating systems and Python versions requires separate evidence.
No Lean sources or proof claims are changed by this implementation.

## Recorded observations

Linux x86-64, Intel Xeon Gold 6458Q, CPython 3.12.3, NumPy 1.26.4, GCC 13.3.0.
Numbers below are median microseconds per call, including the Python API and
result conversion. The legacy column uses already constructed `AbsoluteValue`
objects; their construction is excluded. The fsum column uses an already
constructed Python list; C and NumPy use the same prepared ndarray.

| Input | Elements | Legacy API | C array API | Legacy / C | fsum list | NumPy sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ordinary | 3 | 1.791 | 1.753 | 1.0× | 0.077 | 1.544 |
| cancellation | 3 | 1.815 | 1.731 | 1.0× | 0.077 | 1.532 |
| ordinary | 1,024 | 133.410 | 3.037 | 43.9× | 7.878 | 1.723 |
| cancellation | 1,024 | 127.176 | 2.985 | 42.6× | 4.384 | 1.727 |
| ordinary | 100,000 | 13618.701 | 119.171 | 114.3× | 1509.534 | 16.605 |
| cancellation | 100,000 | 13255.528 | 119.419 | 111.0× | 427.706 | 16.461 |

The native API matched the correctly rounded exact-input reference in these six
samples. This observation is not a universal accuracy guarantee. On the 100,000
element cancellation input, the reference and C returned 33,333; this NumPy
build returned 0. NumPy was faster on large arrays but did not preserve the
residual in this cancellation case. Three-element calls showed essentially no
benefit over the legacy object API. For those small inputs, `math.fsum` on a
prepared list was substantially faster than the full Balansis API.

The standalone C harness passed AddressSanitizer and UBSan. The final full
Python suite passed 757 tests with 103 existing legacy-algebra skips and 83.21%
coverage. An earlier NumPy 2.4.6 run passed 755 tests before two benchmark
provenance tests were added; that NumPy version is outside the current declared
dependency range. A real extension-free run passed 80 focused array tests with
39 native-only skips. Both built wheels were installed in a fresh environment
and exercised outside the checkout, first without and then with the extension.

The native wheel was also built with deliberately unsafe environment values
`CFLAGS=-Ofast LDFLAGS=-ffast-math`: the build's final compiler and linker flags
restored strict arithmetic, and subnormal/cancellation tests passed. Direct
compilation of the kernel with fast-math, bypassing that build configuration,
was rejected by its preprocessor guard. This is tested behaviour on the recorded
GCC environment, not validation of every external toolchain.
