# Array functions and C kernels

`sum_array(values, backend="auto")` returns `(AbsoluteValue, diagnostic)`.
`dot_array(a, b, backend="auto")` returns a float.
`gram_pair(a, b, backend="auto")` returns `(a·a, b·b, a·b)`.

Inputs are one-dimensional real numeric arrays. Paired inputs must have equal
lengths. They are converted to contiguous native float64 buffers. Nonfinite
values raise `ValueError`; overflow raises `OverflowError`.

`sum_array` uses sequential Neumaier accumulation. `dot_array` accumulates exact
products of the represented binary64 inputs and rounds once to nearest-even.
`gram_pair` computes three independently rounded entries in one input pass.
Overflow of any Gram entry raises an error for the operation.

The C implementation stores positive and negative integer accumulators in fixed
arrays. Python uses arbitrary-size integers for the same dot-product contract.
The sum kernel requires nearest-even floating-point rounding with gradual
underflow. The build disables fast-math and floating-point contraction.

`auto` selects an available C entry point. `native` requires that entry point;
`python` selects the reference implementation. Kernel capabilities are reported
by `native_available`, `native_dot_available` and `native_gram_available` in
`balansis.array`.

```python
from balansis import dot_array

assert dot_array([1e308, 1e308], [2., -2.], backend="python") == 0.0
```

The low-level `_balansis_kernels` interface accepts contiguous native float64
buffers, including read-only and unaligned buffers. Calls retain the GIL;
callers must prevent writes from external processes or native threads during a
call. C source: [native/src](../native/src). Build: `python -m pip install ./native`.
