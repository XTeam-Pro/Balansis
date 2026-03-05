[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](https://github.com/XTeam-Pro/Balansis)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Coverage](https://img.shields.io/badge/coverage-95%25%2B-brightgreen.svg)](https://github.com/XTeam-Pro/Balansis)
[![Lean4](https://img.shields.io/badge/Lean4-12%20axioms%20proven-blueviolet.svg)](./formal/)
[![License](https://img.shields.io/badge/license-MIT%20%2F%20Commercial-blue.svg)](./COMMERCIAL_LICENSE.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Balansis

**Python mathematical library implementing Absolute Compensation Theory (ACT) — a numerically stable arithmetic framework that replaces IEEE 754 zero and infinity with structurally sound alternatives.**

[Theory Whitepaper](docs/theory/act_whitepaper.md) | [Changelog](CHANGELOG.md) | [Formal Proofs](formal/) | [tnsim API](tnsim/)

---

## What is ACT?

Absolute Compensation Theory is a mathematical framework that eliminates the root causes of numerical instability in floating-point computation:

- **Replaces zero** with `ABSOLUTE` — an additive identity `AbsoluteValue(magnitude=0.0, direction=1)` that prevents division-by-zero at the type level rather than at runtime.
- **Replaces infinity** with `EternalRatio` — a structurally bounded representation of a ratio whose denominator is guaranteed non-Absolute, making unbounded results impossible to construct.
- **Compensated arithmetic** — every operation in `Operations` returns a `(result, compensation_factor)` tuple so accumulated error is tracked explicitly rather than silently absorbed.
- **Formally verified** — all 12 algebraic axioms are proven in Lean4 (Mathlib v4.28.0) with zero `sorry`, zero errors, and zero admitted axioms.

---

## Installation

```bash
# Core library (Pydantic + NumPy)
pip install balansis

# With specific extras
pip install balansis[plot]      # + matplotlib, plotly
pip install balansis[notebook]  # + jupyter, ipykernel
pip install balansis[torch]     # + torch (EternalTorchOptimizer)
pip install balansis[all]       # everything
```

| Extra | Additional dependencies |
|-------|------------------------|
| `plot` | matplotlib, plotly |
| `notebook` | jupyter, ipykernel |
| `torch` | torch |
| `all` | all of the above |

Core dependencies: `pydantic >= 2.5`, `numpy >= 1.24`. Python 3.10, 3.11, and 3.12 are supported.

---

## Quick Start

### Core types and compensated operations

```python
from balansis import AbsoluteValue, EternalRatio, Operations, Compensator
from balansis import ABSOLUTE, UNIT_POSITIVE, UNIT_NEGATIVE
from balansis import B  # convenience constructor: B(5.0) == AbsoluteValue.from_float(5.0)

# AbsoluteValue: immutable (Pydantic frozen=True), magnitude >= 0, direction in {-1, 1}
a = AbsoluteValue(magnitude=5.0, direction=1)    # +5
b = AbsoluteValue(magnitude=3.0, direction=-1)   # -3

# ABSOLUTE is the additive identity — the ACT replacement for zero
zero = AbsoluteValue(magnitude=0.0, direction=1)  # same as ABSOLUTE

# Inspect values
print(a.to_float())      # 5.0
print(a.is_absolute())   # False
print(a.is_positive())   # True

# Round-trip from Python float
c = AbsoluteValue.from_float(-3.5)  # AbsoluteValue(magnitude=3.5, direction=-1)

# Standard arithmetic operators are overloaded
print(a + b)    # AbsoluteValue(magnitude=2.0, direction=1)   — perfect cancellation
print(a - b)    # AbsoluteValue(magnitude=8.0, direction=1)
print(a * 2.0)  # AbsoluteValue(magnitude=10.0, direction=1)
print(a / 2.0)  # AbsoluteValue(magnitude=2.5, direction=1)
print(-a)       # AbsoluteValue(magnitude=5.0, direction=-1)
print(abs(a))   # AbsoluteValue(magnitude=5.0, direction=1)

# Low-level compensated operations return (result, compensation_factor) tuples
result, comp = Operations.compensated_add(a, b)
result, comp = Operations.compensated_multiply(a, b)
result, comp = Operations.compensated_power(a, 2.0)
result, comp = Operations.compensated_sqrt(a)
result, comp = Operations.compensated_log(a)
result, comp = Operations.compensated_exp(a)

# Kahan-compensated aggregation
total, comp = Operations.sequence_sum([a, b, a])
product, comp = Operations.sequence_product([a, b])

# EternalRatio: structurally safe ratio — denominator cannot be ABSOLUTE
ratio = EternalRatio(numerator=a, denominator=b)
print(ratio.numerical_value())  # -5.0/3.0 (signed float)
print(ratio.is_stable())        # True
simplified = ratio.simplify()

# Division always returns EternalRatio, never raises ZeroDivisionError
ratio = Operations.compensated_divide(a, b)

# High-level Compensator returns AbsoluteValue directly (no tuples)
comp = Compensator()
result = comp.compensate_addition(a, b)        # AbsoluteValue
result = comp.compensate_multiplication(a, b)  # AbsoluteValue
ratio  = comp.compensate_division(a, b)        # EternalRatio
result = comp.compensate_power(a, 2.0)         # AbsoluteValue
```

### Algebraic structures

```python
from balansis.algebra.absolute_group import AbsoluteGroup, GroupElement

# Infinite additive group — identity is ABSOLUTE
add_group = AbsoluteGroup.additive_group()

# Infinite multiplicative group — identity is UNIT_POSITIVE
mul_group = AbsoluteGroup.multiplicative_group()

# Finite cyclic group of given order
cyc_group = AbsoluteGroup.finite_cyclic_group(order=6)

elem_a = GroupElement(value=AbsoluteValue(magnitude=2.0, direction=1))
elem_b = GroupElement(value=AbsoluteValue(magnitude=3.0, direction=1))

result   = add_group.operate(elem_a, elem_b)
identity = add_group.identity_element()
inverse  = add_group.inverse_element(elem_a)
print(add_group.is_abelian())  # True
print(cyc_group.order())       # 6
```

### Linear algebra with ACT compensation

```python
from balansis.linalg.gemm import matmul
from balansis.linalg.qr import qr_decompose
from balansis.linalg.svd import svd

# Matrices are List[List[AbsoluteValue]]
A = [[AbsoluteValue(1.0, 1), AbsoluteValue(2.0, 1)],
     [AbsoluteValue(3.0, 1), AbsoluteValue(4.0, 1)]]

B = [[AbsoluteValue(5.0, 1), AbsoluteValue(6.0, 1)],
     [AbsoluteValue(7.0, 1), AbsoluteValue(8.0, 1)]]

C        = matmul(A, B)     # List[List[AbsoluteValue]]
Q, R     = qr_decompose(A)  # Gram-Schmidt QR decomposition
U, S, Vt = svd(A)           # SVD (requires numpy)
```

### Finance ledger with exact cancellation

```python
from balansis.finance.ledger import Ledger
from decimal import Decimal

ledger = Ledger()
ledger.post_entry("assets",   Decimal("1000.00"), "initial deposit")
ledger.post_entry("assets",   Decimal("500.00"),  "additional funding")
ledger.transfer("assets", "expenses", Decimal("250.00"), "vendor payment")

total = ledger.balance()                   # AbsoluteValue — global balance
assets = ledger.account_balance("assets")  # AbsoluteValue — per-account balance
```

---

## Module Overview

| Module | Import path | Description |
|--------|-------------|-------------|
| Core types | `balansis` | `AbsoluteValue`, `EternalRatio`, `ABSOLUTE`, `UNIT_POSITIVE`, `UNIT_NEGATIVE`, `B` |
| Operations | `balansis` | `Operations` — compensated arithmetic returning `(result, comp)` tuples |
| Compensator | `balansis` | `Compensator` — high-level engine returning `AbsoluteValue` directly |
| Algebra | `balansis.algebra.absolute_group` | `AbsoluteGroup`, `GroupElement` — group theory (axioms A1-A5) |
| Algebra | `balansis.algebra.eternity_field` | `EternityField`, `FieldElement` — field theory (axioms E1-E4, S1-S3) |
| Linear algebra | `balansis.linalg.gemm` | `matmul` — ACT-compensated matrix multiplication |
| Linear algebra | `balansis.linalg.qr` | `qr_decompose` — Gram-Schmidt QR decomposition |
| Linear algebra | `balansis.linalg.svd` | `svd` — Golub-Kahan SVD with NumPy fallback |
| ML optimizer | `balansis.ml.optimizer` | `EternalOptimizer`, `EternalTorchOptimizer` (PyTorch subclass) |
| Sets | `balansis.sets.eternal_set` | `EternalSet` — zero-sum infinite sets |
| Sets | `balansis.sets.generators` | `harmonic_generator`, `grandis_generator` |
| Sets | `balansis.sets.resolver` | `global_compensate`, `verify_zero_sum`, `stream_compensate` |
| Finance | `balansis.finance.ledger` | `Ledger` — double-entry bookkeeping with ACT compensation |
| NumPy | `balansis.numpy_integration` | `to_numpy`, `from_numpy`, `add_arrays` — vectorized bridge |
| Vectorized | `balansis.vectorized` | `batch_add`, `batch_mul_scalar`, `batch_to_float` |
| Arrow | `balansis.arrow_integration` | `to_table`, `from_table` — Apache Arrow integration (requires pyarrow) |
| Pandas | `balansis.pandas_ext` | `AbsoluteValueDtype`, `AbsoluteArray` — pandas extension type (requires pandas) |
| Memory | `balansis.memory.arena` | `AbsoluteArena` — value pool / allocation cache |

---

## Formal Verification

Version 0.2.0 ships a complete Lean4 formalization of ACT using Mathlib v4.28.0. All 12 axioms are machine-checked — **0 `sorry`, 0 errors, 0 admitted axioms**.

| Group | Lean4 file | Proven axioms |
|-------|-----------|---------------|
| AbsoluteGroup | `BalansisFormal/AbsoluteValue.lean` | A1: `add_absolute_right`, A2: `add_comm`, A3: `add_assoc`, A4: `add_inverse`, A5: `add_cancellation` |
| EternityField | `BalansisFormal/EternalRatio.lean` | E1: `mul_identity`, E2: `mul_comm`, E3: `mul_assoc`, E4: `mul_inverse` |
| Cross-structure | `BalansisFormal/Algebra.lean` | S1: `s1_distributivity`, S2: `s2_mul_inverse`, S3: `s3_commutativity_with_add` |
| Direction | `BalansisFormal/Direction.lean` | 13 theorems: `neg_ne_pos`, `double_neg`, `mul_same`, `mul_diff`, and more |

The proofs include `toReal` bridge lemmas connecting ACT types to `ℝ`:

- `AbsoluteValue.toReal`: `toReal (mk m d) = m.toReal * d.toReal`
- `toReal_injective`: structural equality follows from real-number equality
- `EternalRatio.mul_toReal`: multiplication bridge

To verify locally:

```bash
cd formal && lake build
```

---

## Testing

```bash
# Run full test suite with coverage enforcement (>= 95% required)
poetry run pytest

# Run specific modules
poetry run pytest tests/test_absolute.py -v
poetry run pytest tests/test_operations.py -v
poetry run pytest tests/test_algebra.py -v
poetry run pytest tests/test_numpy_integration.py -v
poetry run pytest tests/test_finance.py -v
```

Code quality gates:

```bash
poetry run mypy balansis/        # strict type checking
poetry run black balansis/ tests/
poetry run isort balansis/ tests/
poetry run flake8 balansis/
poetry run pre-commit run --all-files
```

The CI configuration enforces `--cov-fail-under=95` — the build fails if coverage drops below 95%.

---

## tnsim: Zero-Sum Infinite Sets Simulator

`tnsim/` is a standalone FastAPI service for experimenting with zero-sum infinite sets. It is **not included in the pip package** and must be run from the repository.

```bash
uvicorn tnsim.api.main:app --port 8010
```

| Component | Description |
|-----------|-------------|
| `ZeroSumInfiniteSet` | Mathematical implementation of zero-sum infinite sets |
| `parallel_tnsim` | Parallel set operations |
| `tnsim_cache` | Redis-backed result cache |
| REST API | FastAPI endpoints for set management |
| PostgreSQL | Persistent set state storage |

---

## License

Balansis uses **dual licensing**:

- **MIT License** ([LICENSE](LICENSE)) — free for non-commercial use: research, education, personal projects, proof-of-concept work.
- **Commercial License** ([COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md)) — required for commercial use: SaaS products, production deployments, integration into commercial software.

Copyright (c) 2024-2026 Tikhonov Andrey. All rights reserved.

---

Balansis is MAGIC Level 1 (MetaBalansis) in the [StudyNinja-Eco](https://github.com/XTeam-Pro/StudyNinja-Eco) ecosystem — the mathematical foundation on which higher AGI layers are built.
