# Claim Closure Baseline Results

**Audience:** developers, evaluators, maintainers  
**Status:** canonical  
**Artifact type:** measured baseline

This page records the first repository-tracked benchmark and scenario artifact
for public Balansis claims surfaced in `README.md` and the canonical guides.

## Artifact

- generator: `benchmarks/claim_closure_benchmarks.py`
- machine-readable output: `benchmarks/results/claim_closure_baseline.json`

From the repository root:

```bash
PYTHONPATH=. python benchmarks/claim_closure_benchmarks.py
```

The harness refuses an installed package outside this checkout. Its report binds
the Python source fingerprint and Python/NumPy versions, so another editable
installation cannot silently produce evidence for this source tree.

## Covered Scenarios

- large-scale aggregation with cancellation
- exact cancellation of opposite represented operands, checked in both orders
- finance zero-sum balancing
- division contract with explicit denominator guard
- extended division states for `finite`, `infinite`, and `indeterminate` runtime outcomes
- policy-driven singular arithmetic for `raise`, `propagate`, and `saturate` handling
- pipeline-level policy propagation through SVD singular-value telemetry

## Why This Artifact Exists

The goal of this artifact is not to claim universal superiority over every
baseline. Its role is narrower and more important:

- prove that the documented scenarios are runnable
- show the concrete runtime outputs for those scenarios
- expose the tradeoff between structural behavior and execution cost

## Reading The Results

Use this artifact together with:

- [Benchmark Methodology](methodology.md)
- [Result Interpretation](result-interpretation.md)
- [Precision and Stability](../guides/precision-and-stability.md)
- [Scientific Computing](../guides/scientific-computing.md)

## Current Expectations

At minimum, the artifact should show:

- naive float accumulation loses the small residual in the documented large-aggregation case
- Python built-in `sum()` may behave differently across versions and is therefore recorded separately from the naive loop baseline
- Balansis preserves that residual through `Operations.sequence_sum`
- equal opposite operands cancel to zero through `compensated_add`; no ULP is
  invented to replace information rounded away before construction
- the finance helper reaches structural additive identity on a balanced example
- valid division returns an `EternalRatio`, while an `ABSOLUTE` denominator is rejected explicitly
- the extended division path represents `finite / ABSOLUTE` as signed infinity
- the extended division path represents `ABSOLUTE / ABSOLUTE` as indeterminate
- the policy-driven path records telemetry events and can saturate infinite states to finite bounds
- the SVD pipeline records singular-value policy telemetry for zero singular values

## Update Rule

When a public numerical scenario in `README.md` or the guides changes, this
artifact must be regenerated and reviewed in the same change set.
