# Changelog

## [1.2.0]

### Added

- `sum_array`, `dot_array` and `gram_pair` with `auto`, `python` and `native`
  backend selection. The optional `balansis-kernels` 0.3.0 package provides C11
  kernels for compensated summation, exact binary64 dot products and fused
  Gram entries.
- `Ledger.decimal_balance(account=None)` for exact accumulation of submitted
  Decimal amounts, with locally selected precision.
- Native sum, dot and Gram capability reporting in `balansis doctor`.
- API signatures generated from Python source and executable examples in
  scripts, notebooks and guides.
- Distribution checks for missing or outdated Python modules, compiled kernels
  and license files.

### Changed

- Jacobi SVD stores columns contiguously and scales very large or small inputs
  before iteration.
- `CompensatedSum` reduces an array or tensor to one scalar. Tensor reduction
  preserves the autograd graph; `StableSoftmax` normalizes over the last axis.
- TNSIM cache persistence uses JSON and preserves Decimal values as strings.
- TNSIM database configuration honors `DATABASE_URL`, accepts `POSTGRES_*`
  variables and URL-escapes connection credentials.
- Documentation, notebooks and examples describe the current interfaces.
- The documentation workflow checks API signatures and executes examples.

### Fixed

- `AbsoluteValue.exp()` evaluates the signed argument; numeric hashing agrees
  with equality. Division by small scalars avoids reciprocal overflow.
- `EternalRatio.power()` handles negative bases and integer exponent parity.
- Importing arithmetic and ledger modules preserves the caller's Decimal
  context. Ledger transfers retain all digits when that context has low precision.
- Finite cyclic groups use modular addition and inversion. Prime fields use
  modular multiplication and multiplicative inverses.
- Ratio-based polynomial addition preserves cancellation; polynomial
  construction copies the supplied coefficients.
- Infinite stream compensation emits one term per pair, and `batch_add`
  rejects unequal input lengths.
- Jacobi SVD completes zero singular vectors to an orthonormal basis and
  avoids intermediate overflow in rotation calculations.
- Pandas extension-array slicing, missing values, concatenation and indexed
  selection preserve the array interface.
- Plot exports serialize values, ratios and compensation records together.
- TNSIM routes use the implemented service and PostgreSQL schema. Computation,
  validation and compensation responses report calculated results.
- TNSIM cache bounds, executor cleanup, concurrent pool initialization and
  attention gradient propagation.
- TNSIM 1.1.0 distributions include the API package and its schema. Balansis
  wheels include the license-selection and commercial-license files.
- Benchmark imports, numerical references and reported package versions.

## [1.1.0] - 2026-08-31

[PyPI](https://pypi.org/project/balansis/1.1.0/) ·
[Release](https://github.com/StudyLabPro/Balansis/releases/tag/v1.1.0) ·
[Changes](https://github.com/StudyLabPro/Balansis/compare/v1.0.0...v1.1.0)

### Added

- Floating-point primitives in `balansis.core._eft`: `two_sum`, Dekker
  `two_product`, vectorized product splitting, `dot2` and `comp_sum`.
- The `act_jacobi` SVD backend, using one-sided Jacobi rotations and
  compensated Gram inner products.
- Numerical regression tests for summation, dot products and SVD against
  Fraction and NumPy references.

### Changed

- `compensated_dot_product` delegates to `dot2`, which accumulates product
  values and their residual terms with `math.fsum`.
- `svd` accepts a backend selector; `numpy_gesdd` remains the default.

## [1.0.0] - 2026-07-05

[PyPI](https://pypi.org/project/balansis/1.0.0/) ·
[Release](https://github.com/StudyLabPro/Balansis/releases/tag/v1.0.0) ·
[Changes](https://github.com/StudyLabPro/Balansis/compare/v0.5.1...v1.0.0)

### Added

- The `balansis` console command with `--version`, `doctor` and `add`.
- `ExtendedRatio` finite, infinite and indeterminate states, with `raise`,
  `propagate` and `saturate` policies and singular-event telemetry.
- Lean definitions for extended ratios and Python tests of corresponding
  arithmetic and policy cases.
- `CompensatedQRResult` and `CompensatedSVDResult` with diagnostics and tuple
  unpacking. QR supports Householder, Givens and Gram-Schmidt methods.
- `AdaptiveEternalOptimizer` with moment estimates, gradient clipping,
  warmup and cosine learning-rate decay.
- `CompensatedSum`, `StableSoftmax` and `CompensatedMatMul` compatibility
  wrappers.

### Changed

- Package metadata identifies the 1.0.0 stable release.
- Formal definitions reside in `BalansisFormal`, with an `ACT` theorem
  interface and `FormalAudit.lean` checks.
- Release automation builds distributions and checks installation and CLI
  behavior across Python 3.10, 3.11 and 3.12.
- Root licensing files provide the AGPL-3.0 and commercial licensing layout.

## [0.5.1] - 2026-03-05

[PyPI](https://pypi.org/project/balansis/0.5.1/) ·
[Release](https://github.com/StudyLabPro/Balansis/releases/tag/v0.5.1) ·
[Changes](https://github.com/StudyLabPro/Balansis/compare/v0.5.0...v0.5.1)

### Fixed

- Synchronized `balansis.__version__` with the package version.
- Corrected author contact information and the runtime license description.

## [0.5.0] - 2026-03-05

[PyPI](https://pypi.org/project/balansis/0.5.0/) ·
[Release](https://github.com/StudyLabPro/Balansis/releases/tag/v0.5.0) ·
[Changes](https://github.com/StudyLabPro/Balansis/compare/v0.2.0...v0.5.0)

### Added

- `CompensationStrategy.high_precision()`, `balanced()` and `fast()` presets.
- Bounded compensation history, configured through `max_history_size` and
  resized when the strategy changes.
- Linear-algebra and ML benchmark suites, regression tracking and additional
  tests for GEMM, QR, SVD and optimizers.
- Version validation, changelog checks and release-note generation scripts.
- Lean modules under `formal/ACT` for absolute values, algebra and ratios.

### Changed

- `Operations.compensated_divide` returns an `(EternalRatio, compensation)`
  tuple. `Compensator` records the returned compensation value.
- Compensation summaries expose the latest records from the bounded history.

## [0.2.0] - 2026-03-05

[PyPI](https://pypi.org/project/balansis/0.2.0/) ·
[Source](https://github.com/StudyLabPro/Balansis/tree/v0.2.0)

### Added

- Initial PyPI release with `AbsoluteValue`, `EternalRatio`, `Operations` and
  the `Compensator` engine.
- Group and field classes, matrix multiplication, QR decomposition and an
  SVD wrapper.
- Ledger entries and transfers, optimizer classes, stream generators and
  set compensation.
- NumPy, Pandas and Arrow adapters, plotting utilities and a Rust scalar
  extension.
- Lean definitions under `formal/BalansisFormal` and the TNSIM source package.
- Optional dependency groups for plotting, notebooks, PyTorch, Pandas and
  Arrow.
