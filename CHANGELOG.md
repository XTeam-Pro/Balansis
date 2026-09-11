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
