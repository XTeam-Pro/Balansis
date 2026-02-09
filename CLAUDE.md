# CLAUDE.md - Balansis Project Guide

## Project Overview

Balansis is a Python mathematical library (v0.1.0, alpha) implementing **Absolute Compensation Theory (ACT)**. It replaces traditional zero and infinity with mathematically stable "Absolute" and "Eternity" concepts to eliminate numerical instabilities and singularities in computational mathematics.

**Key problem solved:** Division by zero, overflow/underflow, and gradient instability in numerical computing and deep learning.

## Quick Reference

```bash
# Install dependencies
poetry install
poetry install --with dev    # include dev tools

# Run tests (95%+ coverage required)
poetry run pytest

# Formatting & linting
poetry run black balansis/ tests/
poetry run isort balansis/ tests/
poetry run flake8 balansis/ tests/

# Type checking (strict mode)
poetry run mypy balansis/

# Build documentation
cd docs/ && make html

# Pre-commit hooks
poetry run pre-commit install
poetry run pre-commit run --all-files
```

## Repository Structure

```
balansis/                   # Main package
├── core/                   # Core mathematical types
│   ├── absolute.py         # AbsoluteValue: magnitude + direction (replaces zero)
│   ├── eternity.py         # EternalRatio: stable fraction (replaces infinity)
│   └── operations.py       # Compensated arithmetic (add, multiply, divide, power)
├── algebra/                # Algebraic structures
│   ├── absolute_group.py   # Group theory for AbsoluteValue
│   └── eternity_field.py   # Field theory for EternalRatio
├── logic/                  # Compensation engine
│   └── compensator.py      # Compensator with strategies (stability, overflow, etc.)
├── memory/                 # Memory management
│   └── arena.py            # Arena caching for AbsoluteValue pooling
├── ml/                     # Machine learning integration
│   └── optimizer.py        # EternalOptimizer / EternalTorchOptimizer (PyTorch)
├── linalg/                 # Linear algebra with compensation
│   ├── gemm.py             # General matrix multiply
│   ├── svd.py              # Singular value decomposition
│   └── qr.py               # QR decomposition
├── finance/                # Financial computing
│   └── ledger.py           # ACT-based ledger system
├── sets/                   # Set theory
│   ├── eternal_set.py      # Zero-sum set implementation
│   ├── resolver.py         # Global compensation resolution
│   └── generators.py       # Harmonic and Grandis generators
├── utils/                  # Utilities
│   ├── plot.py             # Plotting (currently disabled)
│   └── safe.py             # Safe operation wrappers
├── native/                 # Rust native extensions
├── numpy_integration.py    # NumPy compatibility layer
├── pandas_ext.py           # Pandas extensions
├── arrow_integration.py    # Apache Arrow support
└── vectorized.py           # Vectorized operations

tnsim/                      # TNSIM sub-project (Theory of Zero-Sum Infinite Sets)
├── core/                   # Zero-sum sets, caching, parallelization
├── api/                    # FastAPI application
├── database/               # SQLAlchemy models, migrations
├── integrations/           # Balansis integration
└── tests/                  # Separate test suite (unit, integration, performance)

tests/                      # Main test suite (~673 test functions, 18 files)
docs/                       # Sphinx documentation (API, theory, guides)
examples/                   # Jupyter notebooks
benchmarks/                 # Performance benchmarks
scripts/                    # Build and validation scripts
.github/workflows/          # CI/CD (ci.yml, docs.yml, release.yml)
```

## Tech Stack

- **Language:** Python 3.10+ (supports 3.10, 3.11, 3.12)
- **Build system:** Poetry (poetry-core backend)
- **Core deps:** Pydantic ^2.5, NumPy ^1.24, Matplotlib ^3.7, Plotly ^5.17
- **Testing:** pytest ^7.4 with pytest-cov (95%+ coverage enforced)
- **Type checking:** mypy (strict mode)
- **Formatting:** Black (line-length 88), isort (Black profile)
- **Linting:** flake8 (88 char lines, ignores E203/W503)
- **Docs:** Sphinx with ReadTheDocs theme
- **TNSIM sub-project:** FastAPI, SQLAlchemy, PyTorch, asyncpg

## Code Conventions

### Naming

- **Classes:** PascalCase (`AbsoluteValue`, `EternalRatio`, `Compensator`)
- **Functions/methods:** snake_case (`compensated_add`, `is_absolute`)
- **Constants:** UPPER_SNAKE_CASE (`ACT_EPSILON`, `STABILITY_THRESHOLD`)
- **Files/modules:** snake_case (`absolute_group.py`, `eternal_set.py`)

### Design Patterns

- **Immutable core types:** Pydantic models with `frozen=True` (`AbsoluteValue`, `EternalRatio`)
- **Pydantic v2 validation:** `@validator` decorators for all inputs on core types
- **Full type annotations:** Every function signature, strict mypy compliance
- **Google-style docstrings:** Required for all public methods (90%+ docstring coverage enforced by interrogate)
- **Compensated arithmetic:** Operations return `(result, compensation_factor)` tuples
- **One primary class per core module**

### Key Constants (from `balansis/__init__.py`)

```python
DEFAULT_TOLERANCE = 1e-10
STABILITY_THRESHOLD = 1e-8
MAX_MAGNITUDE = 1e308
MIN_MAGNITUDE = 1e-308
ACT_EPSILON = 1e-15
ACT_STABILITY_THRESHOLD = 1e-12
ACT_ABSOLUTE_THRESHOLD = 1e-20
ACT_COMPENSATION_FACTOR = 0.1
```

### Convenience Function

`B(value)` is a shorthand constructor that converts `int`, `float`, `str`, or `AbsoluteValue` into an `AbsoluteValue` via `AbsoluteValue.from_float()`.

## Core Concepts for AI Assistants

### AbsoluteValue

Replaces zero. Has `magnitude` (float >= 0) and `direction` (float, typically -1 or 1). The identity element `ABSOLUTE` has magnitude 0, direction 1. Created via constructor or `AbsoluteValue.from_float()`.

### EternalRatio

Replaces infinity. A stable fraction composed of two `AbsoluteValue` objects (numerator and denominator). Avoids division-by-zero by design. Has a `unity()` class method for the multiplicative identity.

### Operations

The `Operations` class provides compensated arithmetic: `compensated_add`, `compensated_multiply`, `compensated_divide`, `compensated_power`. Each operation accounts for and returns compensation factors to maintain numerical stability.

### Compensator

The stability engine. Uses `CompensationType` enum (`STABILITY`, `OVERFLOW`, `UNDERFLOW`, `SINGULARITY`, `BALANCE`, `CONVERGENCE`) and configurable `CompensationStrategy` to detect and correct numerical issues.

### CompensationThreshold

Operations use `COMPENSATION_THRESHOLD = 1e-15` to detect when compensation is needed.

## Testing

- **Framework:** pytest with fixtures in `tests/conftest.py`
- **Coverage:** 95% minimum enforced via `--cov-fail-under=95`
- **Test naming:** files `test_*.py`, classes `Test*`, functions `test_*`
- **Key test files:**
  - `test_absolute.py` / `test_eternity.py` - Core type tests
  - `test_operations.py` - Arithmetic operations (largest test file)
  - `test_algebra.py` - Algebraic structure verification
  - `test_compensator.py` - Compensation engine
  - `test_numpy_integration.py` - NumPy interop

Run tests with coverage: `poetry run pytest tests/ -v --cov=balansis`

## CI/CD

GitHub Actions with three workflows:

1. **ci.yml** - Main pipeline (push to main/develop, PRs, daily at 2:00 UTC)
   - Matrix: 3 OS (ubuntu, windows, macos) x 4 Python versions (3.8-3.11)
   - Steps: flake8, black --check, isort --check, mypy, bandit, pytest+coverage
   - Additional jobs: benchmarks (main only), docs build, security scan, SonarCloud, NumPy compatibility (1.20-latest)

2. **docs.yml** - Documentation pipeline (Sphinx build, notebook validation, GitHub Pages deploy)

3. **release.yml** - Release pipeline (build, TestPyPI, PyPI, GitHub Release)

## Pre-commit Hooks

Extensive pre-commit configuration (`.pre-commit-config.yaml`):

- Trailing whitespace, EOF fixer, YAML/TOML/JSON validation
- Black (formatting), isort (imports), flake8 (linting with bugbear/comprehensions/simplify plugins)
- mypy (strict type checking), bandit (security), safety (dependency vulnerabilities)
- pydocstyle (Google convention), interrogate (90%+ docstring coverage)
- codespell (spelling), nbQA (notebook linting), yamllint, markdownlint
- validate-pyproject

## Guidelines for Making Changes

1. **Always run `poetry run pytest`** after making changes to ensure 95%+ coverage is maintained
2. **Run `poetry run mypy balansis/`** - strict mode means all functions need type annotations
3. **Format with Black** before committing (`poetry run black balansis/ tests/`)
4. **Sort imports with isort** (`poetry run isort balansis/ tests/`)
5. **Core types are immutable** - don't try to mutate `AbsoluteValue` or `EternalRatio` instances; create new ones
6. **Follow Google docstring style** for any new public functions/classes
7. **Write tests** for new functionality in the `tests/` directory, following existing patterns
8. **Use Pydantic v2 patterns** for data validation on new models
9. **Compensation operations** should always return both the result and the compensation factor
10. **PlotUtils is currently disabled** - do not import it at the package level

## Related Documentation

- `docs/theory/act_whitepaper.md` - ACT theoretical foundations
- `docs/theory/algebraic_proofs.md` - Algebraic structure proofs
- `docs/guide/precision_and_stability.md` - Precision guide
- `CONTRIBUTING.md` - Contribution guidelines (in Russian)
- `ROADMAP.md` - Development roadmap (v0.2 through v1.0)
- `CHANGELOG.md` - Version history
- `.trae/documents/` - Detailed AI-oriented technical architecture docs
