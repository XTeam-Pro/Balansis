# Integrations

## NumPy

`to_numpy` and `from_numpy` convert between values and a structured array with
`magnitude: float64` and `direction: int8` fields. Object ufuncs dispatch scalar
operations. `compensated_dot_product` flattens its arguments and uses the exact
dot-product implementation. `compensated_softmax` normalizes over the complete
input array. The `StableSoftmax` wrapper normalizes over the last axis.

`CompensatedSum` reduces all input elements to one scalar. Tensor inputs retain
the autograd graph. `CompensatedMatMul` dispatches NumPy, tensor or value-matrix
multiplication according to the input type.

## Pandas and Arrow

Install `.[pandas,arrow]`. `AbsoluteArray._from_sequence` constructs the Pandas
extension array; `AbsoluteValueDtype` registers the `absolute` dtype. Slices,
concatenation, missing values and `take` follow the extension-array interface.
`to_numpy(dtype=float)` converts values and missing entries to floats.

Arrow conversions are `to_record_batch`, `to_table` and `from_table` in
`balansis.arrow_integration`. Columns are named `magnitude` and `direction`.

## PyTorch optimizers

Install `.[torch]`. `EternalOptimizer` normalizes each parameter's update by its
gradient norm and supports momentum and weight decay. `EternalTorchOptimizer`
provides the `torch.optim.Optimizer` interface. `AdaptiveEternalOptimizer` uses
bias-corrected first and second moments, per-parameter gradient clipping, warmup
and optional cosine decay. Singular scale events are exposed on the optimizer.

## Plotting

Install `.[plot,pandas]`. `PlotUtils` supports Matplotlib and Plotly backends,
value and ratio plots, compensation records and sequence animation.
`export_plot_data` writes typed records as JSON, CSV or Excel. Excel export uses
an installed Pandas Excel writer.

## Batch and allocation helpers

`batch_add` requires equally sized lists, `batch_mul_scalar` applies a scalar,
and `batch_to_float` returns a NumPy array. `AbsoluteArena` reuses immutable
values keyed by magnitude and direction. `balansis.native.add_absolute` selects
the optional Rust scalar extension when available.

[API signatures](reference/index.md).
