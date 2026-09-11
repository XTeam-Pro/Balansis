# Balansis C kernels

`balansis-kernels` 0.3.0 provides optional CPython C11 kernels for Balansis:
Neumaier summation, exact binary64 dot products and fused Gram entries.

From this directory, run `python -m pip install .`. From the Balansis repository
root, run `python -m pip install ./native`. The extension module is named
`_balansis_kernels`; its backward-compatible API version is 1.

The public Python entry points are `sum_array`, `dot_array` and `gram_pair` in
`balansis.array`. They select installed C kernels automatically. For example,
`gram_pair([1., 2.], [3., 4.], backend="native")` returns `(5., 25., 11.)`.

C entry points accept one-dimensional contiguous native float64 buffers.
The compiler must support C11 and IEEE-754 binary64. Strict floating-point flags
are applied by `setup.py`. Source files and sanitizer harnesses ship in the sdist.

AGPL-3.0-only or separate commercial license; see the included licensing files.
