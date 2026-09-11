# Balansis C kernels

Optional CPython extension for sequential Neumaier summation of binary64 buffers.
This package is built separately so the Balansis Python package still installs
without a C compiler. No NumPy C ABI dependency, SIMD dispatch or assembly is used.

From the Balansis repository root, in the intended Python environment:

```bash
python -m pip install ./native
python -c "from balansis.array import native_available; assert native_available()"
python -m pytest tests/test_array_sum.py --no-cov
```

Builds require a C11 compiler and Python development headers. GCC/Clang use
`-O3 -fno-fast-math -ffp-contract=off -fexcess-precision=standard`; MSVC uses
`/O2 /fp:strict /std:c11`. Platforms must have binary64 doubles with evaluation
in the declared type. Platform compatibility must be validated before publishing
wheels; the accompanying benchmark records the tested environment.

The kernel keeps the GIL and accepts a contiguous, one-dimensional native double
buffer, including read-only and unaligned buffers. The caller must prevent
concurrent writes by native threads or external processes. It assumes round to
nearest, ties to even, and gradual underflow; altered floating-point modes are
unsupported. The internal API returns `(corrected_sum, signed_correction)`;
Balansis converts this to its public result and diagnostic. API version is 1.

NaN/infinity raise `ValueError`; intermediate, correction or final overflow raises
`OverflowError`. Input is examined in order: the first detected invalid operation
determines the exception. Compensation is not arbitrary precision or a guarantee
of correct rounding. Precision lost before conversion to binary64 is unrecoverable.

The sources are part of Balansis and use its existing dual-license terms; see
`LICENSE`, `NOTICE`, `LICENSING.md` and `COMMERCIAL_LICENSE.md` in this distribution.
