# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Balansis is a Python mathematical library implementing **Absolute Compensation Theory (ACT)** — a novel framework that replaces IEEE 754 zero and infinity with `AbsoluteValue` and `EternalRatio` types for enhanced computational stability. It is **MAGIC Level 1 (MetaBalansis)** in the StudyNinja ecosystem.

## Commands

```bash
# Install dependencies
poetry install

# Run all tests (coverage >= 95% enforced)
pytest

# Run a specific test file
pytest tests/test_absolute.py -v

# Run a single test
pytest tests/test_absolute.py::TestAbsoluteValue::test_addition -v

# Skip slow/performance tests
pytest -m "not slow"

# Run only mathematical property tests
pytest -m mathematical

# Format code
black balansis/
isort balansis/

# Type check (strict mypy)
mypy balansis/

# Lint
flake8 balansis/

# Run all pre-commit hooks
pre-commit run --all-files

# Run benchmarks
python benchmarks/bench_basic.py
```

## Architecture

### Core Types (`balansis/core/`)

**`AbsoluteValue`** (immutable Pydantic model, `frozen=True`):
- `magnitude: float >= 0`, `direction: Literal[-1, 1]`
- `ABSOLUTE = AbsoluteValue(magnitude=0.0, direction=1)` replaces traditional zero
- Perfect cancellation: equal magnitude + opposite direction = ABSOLUTE (no precision loss)
- Operators `+`, `-`, `*`, `/`, `neg` all implemented; `to_float()` and `from_float()` for conversion
- `B(value)` convenience constructor in `balansis/__init__.py`

**`EternalRatio`** (immutable Pydantic model, `frozen=True`):
- `numerator: AbsoluteValue`, `denominator: AbsoluteValue` (denominator cannot be ABSOLUTE — enforced at construction)
- Replaces traditional infinity/division; use `numerical_value()` to get float
- `is_stable()`, `simplify()`, arithmetic operators all implemented

**`Operations`** (static methods only):
- All compensated operations return `(result: AbsoluteValue, compensation_factor: float)` tuples
- Methods: `compensated_add`, `compensated_multiply`, `compensated_divide` (returns `EternalRatio`), `compensated_power`, `sequence_sum` (Kahan), `sequence_product`
- Constants: `COMPENSATION_THRESHOLD = 1e-15`, `MAX_COMPENSATION_ITERATIONS = 100`

### Compensation Engine (`balansis/logic/compensator.py`)

`Compensator` wraps `Operations` with:
- `CompensationStrategy` config (stability/overflow/underflow thresholds)
- Automatic detection of compensation needs (`detect_compensation_need`)
- History tracking via `CompensationRecord` dataclass (for analytics)
- High-level methods: `compensate_addition`, `compensate_multiplication`, `compensate_division`, `compensate_power`

### Algebraic Structures (`balansis/algebra/`)

- `AbsoluteGroup`: Group theory on `AbsoluteValue` — additive (identity = ABSOLUTE) and multiplicative (identity = UNIT_POSITIVE) variants. Supports subgroups, cosets, quotient groups.
- `EternityField`: Field operations on `EternalRatio`. `FieldElement` wraps `EternalRatio`.
- Both use `GroupElement` / `FieldElement` wrapper models.

### Other Modules

- `balansis/linalg/`: `gemm.py` (matrix multiply), `svd.py`, `qr.py` — all operating on `List[List[AbsoluteValue]]`
- `balansis/ml/optimizer.py`: `EternalOptimizer` (generic), `EternalTorchOptimizer` (PyTorch subclass) — ACT-normalized gradient descent
- `balansis/sets/`: `EternalSet` (typed iterable of `AbsoluteValue`), `generators.py` (`harmonic_generator`, `grandis_generator`), `resolver.py` (`global_compensate`)
- `balansis/finance/ledger.py`: Exact-cancellation accounting
- `balansis/memory/arena.py`: Value pooling
- `balansis/numpy_integration.py`: Structured numpy dtype + ufuncs for `AbsoluteValue` arrays
- `balansis/vectorized.py`, `pandas_ext.py`, `arrow_integration.py`: Integration layers

### tnsim (Separate Sub-Application)

`tnsim/` is a FastAPI service (Theory of Null Sum Infinite Sets) that uses balansis. It has its own database (`tnsim/database/`), API routes (`tnsim/api/`), and balansis integration layer (`tnsim/integrations/balansis_integration.py`). It is independent from the main library.

### Formal Proofs (`formal/`)

Lean 4 formal verification of ACT axioms. Entry point: `formal/BalansisFormal.lean` imports `Direction`, `AbsoluteValue`, `EternalRatio`, `Algebra` modules. Build with `lake build` (requires Lean 4 toolchain from `formal/lean-toolchain`).

## Key Design Invariants

1. **Immutability**: `AbsoluteValue` and `EternalRatio` are always immutable. All operations return new instances.
2. **Direction constraint**: `direction` is always exactly `1` or `-1` (not `0`, not `0.5`). The conftest creates some with `0.5` for edge-case testing, but the validator enforces `{-1, 1}`.
3. **No division by Absolute**: `EternalRatio` construction raises `ValueError` if denominator `magnitude == 0.0`. This is the type-level enforcement of division safety.
4. **Compensated ops return tuples**: `Operations.*` methods return `(AbsoluteValue, float)`. The `Compensator` unwraps these and exposes simpler `AbsoluteValue`-returning methods.
5. **Coverage gate**: `pytest` fails if coverage drops below 95%. All new code must be tested.
6. **Mypy strict**: All functions require type annotations; `disallow_untyped_defs = true`.

## Module Constants (from `balansis/__init__.py`)

```python
ABSOLUTE = AbsoluteValue(magnitude=0.0, direction=1)   # zero replacement
UNIT_POSITIVE = AbsoluteValue(magnitude=1.0, direction=1)
UNIT_NEGATIVE = AbsoluteValue(magnitude=1.0, direction=-1)
ACT_EPSILON = 1e-15           # numerical comparison epsilon
ACT_STABILITY_THRESHOLD = 1e-12
ACT_COMPENSATION_FACTOR = 0.1
```

## Test Markers

| Marker | Usage |
|--------|-------|
| `slow` | Performance/large-data tests (`-m "not slow"` to skip) |
| `integration` | End-to-end tests |
| `mathematical` | Axiom and property tests |
| `plotting` | Tests requiring matplotlib/plotly |

Custom assertions in `tests/conftest.py`: `assert_act_equal`, `assert_ratio_equal`.
