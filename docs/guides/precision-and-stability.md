# Precision and Stability

**Audience:** developers, researchers, evaluators  
**Status:** canonical

This guide explains where standard floating-point arithmetic becomes misleading
and how Balansis exposes compensation as part of the runtime model.

## When To Read This

- you are investigating unstable reductions
- you need to explain why Balansis differs from plain IEEE 754 results
- you want a canonical entry point for benchmark and accuracy discussions

## Current Navigation

- product-level motivation: [Why Balansis](../getting-started/why-balansis.md)
- runnable examples: [Examples](../examples/index.md)
- benchmark entry point: [Benchmarks](../benchmarks/index.md)
- mathematical context: [Mathematics](../mathematics/index.md)

## Represented Values and Cancellation

`compensated_add` operates on the magnitudes actually stored in its operands.
Equal magnitudes with opposite directions produce `ABSOLUTE` at every scale.
For distinct near-equal large magnitudes, the represented difference is kept.
Swapping those operands preserves the numerical result.

An earlier implementation substituted a signed ULP when equal magnitudes were
above `9e15`. It could return `+2` for `1e16 + (-1e16)` and `-2` in the reverse
order. That behavior is corrected: a guessed residual is not numerical evidence.
Callers relying on a nonzero value to avoid division by zero must handle the
denominator explicitly, using the finite or extended division contract.

Information rounded away before `AbsoluteValue.from_float` is called cannot be
recovered by addition. For example, `float(1e16 + 1.0)` is already `1e16`.
Passing the three original terms to `sequence_sum` instead preserves the small
term using Neumaier summation. Python's built-in `sum` is version-dependent;
benchmark it separately from a naive loop and include `math.fsum` as a baseline.

## Compensation Factor

The second return value has operation-specific diagnostic semantics. It is not
uniformly an additive correction, relative error, or rigorous uncertainty bound.
For `compensated_add`, exact opposite nonzero operands retain the legacy
`compensation_factor * STABILITY_FACTOR` diagnostic. Nonzero near-cancellation
uses the magnitude-to-difference ratio; the standard path returns `1.0`, and
the small-result path reports its threshold ratio. These diagnostics must not
be added to the numerical result or interpreted as recovered information.

`sequence_sum` instead reports the absolute accumulated correction divided by
its compensation threshold. Its numerical result already includes that correction.

The Lean model and float runtime have different representations. This change
establishes the tested cancellation contract; it does not claim a proof of all
floating-point operations or a universal error bound.
