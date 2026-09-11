# Fused exact Gram pairs and column-contiguous Jacobi SVD

The previous native SVD evaluated `dot(a,a)`, `dot(b,b)` and `dot(a,b)` in three
Python/C calls for every pair of columns. Each `dot2` call also checked inputs
and flattened columns from the row-contiguous working matrix, copying strided
data. The new path obtains all three entries in one native call, and keeps the
working matrix in Fortran order so columns are contiguous views.

## API and contract

```python
from balansis import gram_pair

assert gram_pair([1, 2], [3, 4], backend="native") == (5.0, 25.0, 11.0)
```

The result order is `(dot(a,a), dot(b,b), dot(a,b))`. Finite real 1-D inputs must
have equal lengths. Other numeric dtypes are converted to float64, potentially
rounding the inputs before the exact products are formed. Each exact sum is
rounded once to binary64, with ties to even. A rounded overflow in any entry
raises `OverflowError`, even if the cross product alone would be finite.
NaN/infinity anywhere in either input raises `ValueError` before norm overflow.
No partial tuple is returned on failure.

The C kernel decodes each pair of elements once and accumulates both squares
and their cross product. The existing exact-product multiplication, accumulator
bound and final rounding code are reused. Four magnitude accumulators and a zero
scratch array occupy 2720 bytes, independent of vector length. This extends the
[exact-dot implementation](native-dot.md); it does not introduce a new rounding
algorithm or change the existing sum/dot contracts. No new Lean proof is claimed.

Install native extension 0.3 from the source checkout with
`python -m pip install ./native`. `native_gram_available()` reports support.
`backend="native"` requires the fused kernel. Auto mode can use three exact-dot
calls with a 0.2 extension, or the integer Python reference without C. A 0.1
extension still accelerates sums. Existing `balansis-python` users receive the
new path when their installed Python/native wheel pair is updated together.

## SVD integration

Jacobi SVD stores its private working copy in Fortran order, including the wide
matrix transpose path. The three rotation-driving entries come from `gram_pair`;
final column norms still use `dot2`. Rotation formulas, tolerance, sweep limit,
sorting and output construction are unchanged. Input matrices remain unmodified.

Regression tests compare all returned arrays byte for byte against the frozen
pre-fusion function from commit `d2f4e143`. Its AST was checked against that
commit. Cases include tall, wide, single-column, zero, rank-deficient and
ill-conditioned matrices. Instrumented native calls verify both columns share
the same Fortran-contiguous working matrix and arrive as contiguous views,
without an intervening input copy.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  python benchmarks/native_gram_benchmarks.py \
  --output benchmarks/results/native_gram_comparison.json
python -m pytest tests/
```

The [JSON artifact](../../benchmarks/results/native_gram_comparison.json) records
source/binary hashes, Python and NumPy versions, CPU, BLAS thread settings and
timing distributions. It refuses an unrelated Balansis import or stale native
binary. Inputs are synthetic (seed 774), prepared outside the timed region.
Nine calibrated, shuffled rounds run with cyclic GC disabled. The host is shared;
CPU frequency and affinity are not controlled.

The baseline already uses exact C dot products. Complete SVD timing includes
all copies, rotations, convergence checks and outputs. A separate layout-only
variant keeps the new column layout but restores three `dot2` calls; this helps
distinguish the effects of layout and fusion. All variants must return bitwise
identical decompositions before timings are reported. The older native-dot
benchmark now uses the frozen pre-fusion function, so replacing its `dot2` still
measures the historical comparison it describes.

## Verification

The full Python suite passed 912 tests with 103 existing legacy algebra skips;
coverage was 83.34%. A real extension-free Gram test run passed 37 tests and
skipped 18 native-only cases. New checks include independent `Fraction` oracles,
random exponent ranges, permutation invariance, signed underflow, norm overflow,
nonfinite precedence, buffer validation, export release on errors and compatibility
with old extensions. The standalone C Gram harness passed AddressSanitizer and
UndefinedBehaviorSanitizer, including unaligned buffers and 100,000-element inputs.

```bash
cc -std=c11 -O2 -fno-fast-math -ffp-contract=off \
  -fsanitize=address,undefined -fno-omit-frame-pointer -I native/src \
  native/src/exact_dot.c native/tests/test_gram_pair.c -o /tmp/balansis-gram-check
/tmp/balansis-gram-check
```

The numerical argument remains an implementation argument backed by tests, not
formal verification of the C program. Results apply to the measured environment;
other compilers, Python versions and operating systems need separate validation.

## Recorded measurements

CPython 3.12.3, NumPy 1.26.4, Linux x86-64, Intel Xeon Gold 6458Q;
BLAS thread limits set to one. Complete SVD medians in milliseconds:

| Matrix | Previous C dot SVD | Column layout only | Fused Gram SVD | Additional speedup |
| --- | ---: | ---: | ---: | ---: |
| 16 × 4 | 0.610 | 0.582 | 0.276 | 2.21× |
| 64 × 8 | 4.019 | 3.776 | 1.770 | 2.27× |
| 256 × 8 | 5.329 | 4.817 | 2.632 | 2.02× |
| 8 × 64 | 4.001 | 3.863 | 1.729 | 2.31× |
| 1024 × 8 | 8.285 | 7.089 | 4.447 | 1.86× |

All compared decompositions were bitwise identical. This is additional speedup
over the already accelerated exact-C-dot version, not the original Python
implementation. Fusion accounts for most of the gain; layout alone helps less.

Gram entries, microseconds per public API call:

| Elements | Three integrated dot2 calls | Fused Gram API | Speedup |
| --- | ---: | ---: | ---: |
| 3 | 11.448 | 1.640 | 6.98× |
| 1,024 | 31.761 | 14.330 | 2.22× |
| 100,000 | 2175.648 | 1575.312 | 1.38× |

The benefit is largest for short vectors, where call/validation overhead matters
most. Large vectors benefit more modestly because all three exact products still
require integer accumulation. No SIMD-specific or handwritten assembly kernel
was introduced.
