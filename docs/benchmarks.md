# Benchmarks

Run benchmarks from the repository root after installing the package and C
extension from that checkout:

```bash
python benchmarks/native_sum_benchmarks.py
python benchmarks/native_dot_benchmarks.py
python benchmarks/native_gram_benchmarks.py
python benchmarks/claim_closure_benchmarks.py
```

The native harnesses check package location and source fingerprints before
measurement. They use seeded synthetic inputs, prepare inputs outside timed
sections, shuffle execution order and report calibrated repeated measurements.
Reports identify Python, NumPy, compiler settings and implementation hashes.

Sum measurements compare the array API with prebuilt `AbsoluteValue` objects.
Dot measurements compare exact accumulation with a frozen arithmetic baseline.
Gram measurements compare three dot calls, contiguous-column layout and a fused
Gram call. SVD comparisons measure complete decompositions.

`AccuracyBenchmark` computes a rational reference from represented float inputs.
`PerformanceBenchmark`, `LinalgBenchmark` and `MLBenchmark` provide configurable
experiments. `RegressionTracker` compares named metrics with a supplied baseline.
The ML harness contains its own NumPy optimizers.

[Harnesses and recorded data](../benchmarks).
