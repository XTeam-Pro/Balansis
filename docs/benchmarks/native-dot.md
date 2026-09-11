# Exact native dot products

`dot_array(a, b, backend="auto")` computes the exact sum of products of the
represented binary64 inputs, followed by one round-to-nearest-even conversion
to binary64. It is implemented with integer arithmetic, not by substituting
Neumaier summation for the previous product-error recovery algorithm.

## Use and integration

Install the optional extension from the source checkout:

```bash
python -m pip install ./native
python -c "from balansis.array import native_dot_available; assert native_dot_available()"
```

```python
import math
from balansis import dot_array

tiny = math.ulp(0.0)
assert dot_array([tiny, tiny], [0.5, 0.5], backend="native") == tiny
assert dot_array([1e308, 1e308], [2.0, -2.0], backend="native") == 0.0
```

Both inputs must be finite real 1-D arrays of equal length. Numeric inputs are
converted to float64 first; that conversion can lose input precision. Contiguous
native float64 arrays need no input copy. Strided or non-native arrays are
converted. NaN/infinity raise `ValueError`. Only a final rounded result outside
float64 range raises `OverflowError`: overflowing products may cancel exactly.
Exact zero returns +0.0; a negative nonzero result below the rounding threshold
returns -0.0. Native buffers may be read-only or unaligned. The GIL is retained;
callers must exclude external/native concurrent writes.

The optional native distribution is version 0.2.0. Its sum API remains compatible
with version 0.1.0. `native_dot_available()` detects the extra operation separately.
`backend="python"` runs the integer reference; `backend="native"` requires the new
extension. Auto mode falls back to the reference if the operation is unavailable.

`core._eft.dot2` now delegates finite inputs to this exact implementation. Existing
`numpy_integration.compensated_dot_product` and `_act_jacobi_svd` therefore use
the native kernel automatically when installed. Their flattened-input interface
is preserved; nonfinite inputs keep the NumPy propagation path. Mismatched
flattened lengths now raise `ValueError`, including empty/nonempty mismatches.
The previous broadcasting behaviour could compute an unintended expression.

The exact Python reference can be slower than the old compensated algorithm on
large inputs. Acceleration requires the native extension. The reference retains
the same numerical contract instead of silently losing accuracy when C is absent.

## Why the previous path needed a stronger boundary

The previous implementation formed rounded products, recovered errors using
Dekker splitting, and called `math.fsum` over concatenated high/low arrays.
Splitting can overflow, and an exact product residual can be too small for a
binary64 value. A finite rounded product alone does not guarantee exact recovery.

Directly reproduced counterexamples:

| Inputs | Previous `dot2` | Exact reference and new implementation |
| --- | ---: | ---: |
| Two smallest subnormals, each multiplied by 0.5 | 0 | Smallest positive subnormal |
| `[1e308, 1e308]` dotted with `[2, -2]` | NaN | 0 |

These are numerical defects in the former full-range claim, not evidence against
the classical transforms within their valid domains. The range conditions matter
in the [Ogita–Rump–Oishi paper](https://ogilab.w.waseda.jp/ogita/math/doc/2005_OgRuOi.pdf).

## Accumulator and rounding argument

Every finite binary64 number can be expressed as an integer significand times
a power of two. Every product is therefore an integer multiple of `2^-2148`,
with magnitude strictly below `2^2048`. A valid input count fitting a 64-bit
`size_t` contains fewer than `2^64` terms. Each sign's accumulated magnitude
thus needs at most 4260 bits in units of `2^-2148`.

The C kernel allocates separate positive and negative arrays of 136 unsigned
32-bit limbs: 4352 bits each, 1088 bytes total, independent of input length.
It multiplies the two 53-bit significands using 32-bit partial products, shifts
the exact product to its exponent position, and propagates integer carries.
Unsigned arithmetic and the accumulator bound avoid signed-overflow assumptions.

After subtracting the smaller sign accumulator, the kernel locates the top bit,
keeps the 53 normal significand bits or the subnormal quantum, and uses guard,
sticky and parity bits for ties-to-even rounding. The final binary64 encoding is
constructed directly. The Python reference instead uses arbitrary-size integers
and integer true division. Tests use a third path: a `Fraction` sum over the
original represented inputs and their exact products.

This is an implementation argument plus executable tests, not a Lean proof of
the C program. Exact accumulators are an established technique; see the
[ExBLAS presentation](https://www.nist.gov/system/files/documents/itl/ssd/is/NRE-2015-04-iakymchuk.pdf).
No third-party kernel implementation was copied. No existing Lean sources or
theorem claims are changed.

## Reproduce and validate

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  python benchmarks/native_dot_benchmarks.py \
  --output benchmarks/results/native_dot_comparison.json
python -m pytest tests/
```

The harness checks Python source location and the native binary's embedded source
hash. It records Python/NumPy versions, CPU, thread settings and binary/source
hashes. Synthetic inputs use seed 840. Exact rational oracles run outside timing.
Nine warmed, calibrated rounds shuffle method order with cyclic GC disabled.
Inputs are prepared arrays; public API measurements include validation and return
handling. `fsum_rounded_products` is explicitly a comparison over already rounded
products, not an exact-product oracle. NumPy/BLAS is a throughput baseline with a
different accuracy contract.

The SVD comparison replaces only the dot function inside the same Jacobi
implementation. It includes matrix copies, rotations, convergence checks and
output construction. Reconstruction and orthogonality errors accompany timings.
This is one shared host with no CPU pinning or frequency control; speedups are
observations, not cross-platform guarantees.

Tests exercise all binary64 exponent positions with exact ties, tie-breaking
sticky bits and negative cases; random full-bit-pattern inputs; permutations;
cancellation across the exponent range; overflow; subnormal rounding; buffer
validation and release on exceptions; old sum-only extensions; and SVD dispatch.
Run the focused tests without the extension as well to exercise actual fallback.

The standalone kernel can be checked with sanitizers:

```bash
cc -std=c11 -O2 -fno-fast-math -ffp-contract=off \
  -fsanitize=address,undefined -fno-omit-frame-pointer -I native/src \
  native/src/exact_dot.c native/tests/test_exact_dot.c -o /tmp/balansis-dot-check
/tmp/balansis-dot-check
```

Build/install both wheel distributions in a separate environment and exercise
them outside the checkout. Other Python versions, compilers and operating systems
still require dedicated compatibility checks.

## Recorded results

CPython 3.12.3, NumPy 1.26.4, Linux x86-64, Intel Xeon Gold 6458Q.
OPENBLAS/OMP/MKL thread limits were set to 1. Full measurements, including
Python fallback and NumPy comparisons, are in the [JSON artifact](../../benchmarks/results/native_dot_comparison.json).

Dot-product medians in microseconds per call:

| Input | Elements | Legacy dot2 | New array API | Speedup | Integrated dot2 | Integrated speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ordinary | 3 | 9.295 | 1.050 | 8.85× | 3.763 | 2.47× |
| cancellation | 3 | 9.187 | 1.012 | 9.08× | 3.706 | 2.48× |
| ordinary | 1,024 | 95.350 | 6.568 | 14.52× | 9.825 | 9.70× |
| cancellation | 1,024 | 78.039 | 6.364 | 12.26× | 9.691 | 8.05× |
| ordinary | 100,000 | 11380.989 | 618.072 | 18.41× | 658.133 | 17.29× |
| cancellation | 100,000 | 9552.740 | 520.289 | 18.36× | 558.609 | 17.10× |

Jacobi SVD medians in milliseconds per call, including the whole decomposition:

| Matrix | Legacy dot backend | Native dot backend | Speedup | Relative reconstruction error |
| --- | ---: | ---: | ---: | ---: |
| 16 × 4 | 1.233 | 0.606 | 2.04× | 4.51e-16 |
| 64 × 8 | 10.109 | 3.909 | 2.59× | 9.8e-16 |
| 256 × 8 | 20.725 | 5.094 | 4.07× | 1.08e-15 |

All six measured native dot-product samples matched the rounded rational
reference. The boundary examples also matched, including the two defects in the
previous implementation. NumPy/BLAS remained faster; the exact-dot implementation
should be chosen when exact accumulation is part of the required contract.
The Python exact backend was slower than the old algorithm on large vectors.

Validation: 857 tests passed, 103 existing algebra tests skipped; Python coverage
83.12%. A real extension-free focused run passed 146 tests with 73 native-only
skips. The new rounding test checks 6,294 tie/sticky/negative examples per backend
across all binary64 exponent positions. The standalone C kernel passed ASan and
UBSan. Benchmark-baseline executable AST matches the previous `dot2`; no SVD
rotation or convergence algorithm was substituted in the performance comparison.

Built wheels were installed in a fresh environment outside the source checkout:
the Python-only exact path passed; installing the previous 0.1 native wheel kept
C sums working with exact Python dot fallback; upgrading to the 0.2 wheel enabled
native exact dot products and passed a complete Jacobi SVD reconstruction check.
Source-distribution contents, Python wheel source contents and unchanged canonical
license files were also verified.
