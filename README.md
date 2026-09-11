[![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](https://github.com/StudyLabPro/Balansis)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Coverage](https://img.shields.io/badge/coverage-85%25%2B-brightgreen.svg)](https://github.com/StudyLabPro/Balansis)
[![Lean4](https://img.shields.io/badge/Lean4-A1--A5%2C%20E1--E4%2C%20S1--S3%20proved-blueviolet.svg)](./formal/)
[![License](https://img.shields.io/badge/license-AGPL--3.0%20%2F%20Commercial-blue.svg)](./LICENSING.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Balansis

**Balansis is a scientific and engineering Python library for workloads where
plain floating-point arithmetic hides instability instead of making it explicit.**

It is built around **Absolute Compensation Theory (ACT)**: a numerical model
that introduces structured runtime objects such as `AbsoluteValue`,
`ABSOLUTE`, and `EternalRatio`, and exposes compensation directly in the API
instead of silently discarding it.

[Documentation](docs/index.md) | [Quick Start](docs/getting-started/quickstart.md) | [API Reference](docs/api/index.md) | [Formal Verification](docs/formal/overview.md) | [Examples](examples/README.md) | [Benchmarks](benchmarks/README.md)

---

## What Problem Does Balansis Solve?

IEEE 754 floating-point arithmetic is fast and ubiquitous, but several failure
patterns still matter in real systems:

- large reductions lose small but meaningful residuals
- catastrophic cancellation turns uncertain differences into misleading zeros
- divide-by-zero handling is deferred to edge-case logic instead of the data model
- financial and simulation pipelines often hide correction logic outside the core arithmetic API

Balansis is for cases where you want those edge conditions to be **visible,
structured, and auditable**.

## Why Was It Created?

Balansis was created to explore a stronger arithmetic interface for:

- scientific computations with unstable reductions
- long-running simulations with drift-sensitive accumulation
- financial workflows that benefit from structural cancellation semantics
- research pipelines that want a bridge from runtime code to formal proofs

---

## Real-World Value First

### 1. Large-Scale Aggregation

```python
total = 0.0
for value in [1e16, 1.0, -1e16]:
    total += value
total  # naive float accumulation
# 0.0
```

```python
from balansis import AbsoluteValue as Bv, Operations

values = [
    Bv.from_float(1e16),
    Bv.from_float(1.0),
    Bv.from_float(-1e16),
]
result, compensation = Operations.sequence_sum(values)
# result == AbsoluteValue.from_float(1.0)
# compensation > 0.0
```

Why it matters: the small residual survives, and the API exposes that a
meaningful correction was applied.

### 2. Catastrophic Cancellation

```python
(1e16 + 1.0) - 1e16  # IEEE 754 / Python float
# 0.0
```

```python
from balansis import AbsoluteValue as Bv, Operations

a = Bv.from_float(1e16)
b = Bv.from_float(-1e16)
result, compensation = Operations.compensated_add(a, b)
# result.is_absolute() == True
```

Equal represented magnitudes with opposite signs cancel exactly, including at
large magnitudes. Balansis cannot recover the `1.0` already lost in
`1e16 + 1.0` before construction. Keep separate terms and use `sequence_sum`
when their small residual must survive. The compensation factor is a diagnostic,
not a replacement value or a rigorous error bound. See
[Precision and Stability](docs/guides/precision-and-stability.md).

### 3. Financial Cancellation

```python
from decimal import Decimal
from balansis.finance.ledger import Ledger

ledger = Ledger()
ledger.post_entry("cash", Decimal("250.00"))
ledger.post_entry("cash", Decimal("-250.00"))

balance = ledger.balance()
# balance == ABSOLUTE
```

Why it matters: offsetting entries cancel structurally to the ACT identity,
which is easier to reason about than burying bookkeeping semantics in plain
float totals.

### 4. Division Edge Handling

```python
1.0 / 0.0  # standard runtime edge case
# ZeroDivisionError
```

```python
from balansis import AbsoluteValue, Operations

num = AbsoluteValue.from_float(6.0)
den = AbsoluteValue.from_float(2.0)
ratio, compensation = Operations.compensated_divide(num, den)
```

Why it matters: Balansis makes ratio structure explicit through `EternalRatio`
for valid denominators and rejects an `ABSOLUTE` denominator directly instead of
pretending that infinity-like behavior is a normal value.

---

## Why Adopt Balansis?

- **Explicit compensation:** low-level operations return both a result and a compensation factor
- **Structured edge handling:** ratio and additive-identity behavior are part of the model, not scattered ad hoc checks
- **Research continuity:** the repository includes a Lean4 formal layer for the public ACT theorem surface
- **Practical scope:** core arithmetic, algebraic structures, linear algebra, finance helpers, and experimental subprojects

---

## Install In 60 Seconds

```bash
pip install balansis
```

CLI / pipx install:

```bash
pipx install balansis
balansis --version
balansis doctor
```

Optional extras:

```bash
pip install balansis[plot]
pip install balansis[notebook]
pip install balansis[torch]
pip install balansis[all]
```

Supported Python versions: 3.10, 3.11, 3.12.

More install options: [Installation Guide](docs/getting-started/installation.md)

---

## Quick Start

```python
from balansis import AbsoluteValue, Operations, ABSOLUTE

a = AbsoluteValue(magnitude=5.0, direction=1)
b = AbsoluteValue(magnitude=3.0, direction=-1)

result, compensation = Operations.compensated_add(a, b)

print(result)
print(compensation)
print(ABSOLUTE)
```

Continue with:

- [Quick Start](docs/getting-started/quickstart.md)
- [Glossary](docs/glossary.md)
- [Examples](examples/README.md)

---

## Batch summation with an optional C kernel

```python
import numpy as np
from balansis import sum_array

values = np.array([1e16, 1.0, -1e16], dtype=np.float64)
result, diagnostic = sum_array(values)
assert result.to_float() == 1.0
```

`sum_array` uses sequential Neumaier compensation. It accepts one-dimensional
real numeric arrays and returns `(AbsoluteValue, diagnostic)` with the same
diagnostic scaling as `Operations.sequence_sum`. The diagnostic is not an error
bound; compensation does not guarantee correct rounding for every input.

The Python implementation is always available. To enable the optional C kernel,
install it from this source checkout with `python -m pip install ./native`.
`backend="native"` requires that kernel; `backend="python"` selects the reference;
the default `backend="auto"` selects an available compatible kernel. A contiguous
native `float64` NumPy array is passed without an input copy. Other real numeric
inputs are converted, which can round integers or higher-precision values.
NaN/infinity and intermediate overflow are rejected. Existing object-based
`Operations.sequence_sum` behaviour is unchanged.

See [native build and numerical contract](native/README.md) and
[reproducible performance comparison](docs/benchmarks/native-sum.md).

### Exact dot products

```python
from balansis import dot_array

assert dot_array([1e308, 1e308], [2.0, -2.0]) == 0.0
```

`dot_array(a, b, backend="auto")` sums the products of represented float64
values exactly and rounds once to float64. Inputs are real, finite 1-D arrays
of equal length. Final rounded overflow raises `OverflowError`. The optional
native package version 0.2 supports this operation; otherwise the integer Python
reference is used. That reference can be slower than the previous compensated
algorithm, so acceleration requires installing the native extension.

Existing `compensated_dot_product` and Jacobi SVD use this path through `dot2`.
They keep their flattened-input interface and legacy NumPy propagation for
nonfinite inputs. Mismatched flattened lengths now raise `ValueError` instead
of allowing unintended broadcasting. See the
[exact-dot implementation and measurements](docs/benchmarks/native-dot.md).

### Fused Gram pairs for Jacobi SVD

`gram_pair(a, b)` returns `(dot(a,a), dot(b,b), dot(a,b))`, with each exact sum
rounded once to float64. Native extension 0.3 computes all three entries in one
input pass. Jacobi SVD uses this operation automatically and stores its working
matrix by columns, avoiding copies of each column at the Python/C boundary.
The rotation formulas and convergence criteria are unchanged.

Older extensions and the Python reference remain supported. See the
[Gram-pair measurements and compatibility checks](docs/benchmarks/native-gram.md).

## Documentation By Audience

| Audience | Start here | Why |
|----------|------------|-----|
| Decision makers | [Why Balansis](docs/getting-started/why-balansis.md) | Understand the problem, the value proposition, and where Balansis fits |
| Developers | [Quick Start](docs/getting-started/quickstart.md) | Install the package and start using the runtime surface |
| Researchers | [Mathematics](docs/mathematics/index.md) | Explore ACT concepts, notation, and theorem-oriented material |
| Verification-oriented readers | [Formal Verification](docs/formal/overview.md) | Review the Lean architecture and current proof status |
| Contributors | [Contributing](CONTRIBUTING.md) | Set up the repo, quality gates, and documentation workflow |

## Documentation Map

- [Documentation Home](docs/index.md)
- [Getting Started](docs/getting-started/why-balansis.md)
- [Core Concepts](docs/concepts/index.md)
- [Architecture](docs/architecture/repository-map.md)
- [API Reference](docs/api/index.md)
- [Mathematics](docs/mathematics/index.md)
- [Formal Verification](docs/formal/overview.md)
- [Examples](docs/examples/index.md)
- [Benchmarks](docs/benchmarks/index.md)
- [TNSIM Overview](docs/tnsim/overview.md)
- [Research Materials](docs/research/index.md)

---

## Formal Verification Status

Version `1.1.0` ships a compiled Lean4 formalization on Mathlib `v4.28.0`.

- `BalansisFormal` is the constructive core
- `ACT` is the public theorem facade
- `formal/` contains **0 `axiom`, 0 `sorry`, 0 `admit`**
- public theorem groups A1-A5, E1-E4, and S1-S3 are compiled as Lean theorems

Verification entrypoints:

- [Formal Overview](docs/formal/overview.md)
- [Lean README](formal/README.md)

---

## Package Surface

Core areas currently present in the repository:

- `balansis.core`
- `balansis.algebra`
- `balansis.linalg`
- `balansis.finance`
- `balansis.numpy_integration`
- `balansis.ml`
- `balansis.sets`

Reader-oriented API navigation starts here: [API Reference](docs/api/index.md)

---

## TNSIM

`tnsim/` is a repository subproject for zero-sum infinite sets experimentation.
It is maintained in the same repository, but it is **not** the main `balansis`
package entrypoint.

- [TNSIM Overview](docs/tnsim/overview.md)
- [TNSIM README](tnsim/README.md)

---

## Contributing

Contributions are welcome, but Balansis is maintained under a dual-license
model. Before opening a substantial pull request, read:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CLA.md](CLA.md)
- [LICENSING.md](LICENSING.md)
- [NOTICE](NOTICE)
- [SECURITY.md](SECURITY.md)
- [Documentation Standards](docs/standards.md)

---

## License

Balansis is dual-licensed:

- [LICENSE](LICENSE): AGPL-3.0
- [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md): commercial proprietary terms
- [LICENSING.md](LICENSING.md): routing and selection guide

Commercial execution material:

- [ORDER_FORM_TEMPLATE.md](ORDER_FORM_TEMPLATE.md)
- [SECURITY.md](SECURITY.md)

Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.

---

Balansis is part of the [StudyNinja-Eco](https://github.com/StudyLabPro/StudyNinja-Eco) ecosystem.
