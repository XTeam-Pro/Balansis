# Changelog

All notable changes to the Balansis project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Balansis v0.5 (Phase 8 target): stable API, complete `linalg/` (GEMM, SVD, QR), PyTorch integration
- ACT benchmark suite vs IEEE 754 and Kahan summation
- LaTeX paper draft for arxiv submission

---

## [0.2.0] — 2026-02-18 — Lean4 Formal Verification Edition

### Added

#### Lean4 Formal Proofs (`formal/BalansisFormal/`)
Complete formalization of ACT using mathlib4 (v4.28.0). All 12 axioms proven — **0 sorry, 0 axioms, 0 errors**.

- **`Direction.lean`** (77 lines)
  - `Direction` type: `Pos | Neg`
  - 13 theorems: `neg_ne_pos`, `double_neg`, `mul_same`, `mul_diff`, `eq_or_ne`, etc.

- **`AbsoluteValue.lean`** (305 lines)
  - `AbsoluteValue` over ℝ (NNReal magnitude + Direction)
  - `toReal` bridge: `toReal (mk m d) = m.toReal * d.toReal`
  - `toReal_injective`: structural equality from real equality
  - Axioms proven as theorems:
    - **A1** `add_absolute_right`: additive identity (`ABSOLUTE` = zero analog)
    - **A2** `add_comm`: commutativity
    - **A3** `add_assoc`: associativity
    - **A4** `add_inverse`: additive inverse exists (opposite direction, equal magnitude)
    - **A5** `add_cancellation`: perfect cancellation (equal magnitude, opposite direction → ABSOLUTE)

- **`EternalRatio.lean`** (208 lines)
  - `EternalRatio` type (replaces IEEE 754 infinity)
  - `mul_toReal` bridge proof
  - Axioms proven as theorems:
    - **E1** `mul_identity`: multiplicative identity
    - **E2** `mul_comm`: commutativity
    - **E3** `mul_assoc`: associativity
    - **E4** `mul_inverse`: multiplicative inverse exists

- **`Algebra.lean`** (192 lines)
  - `mulInv`: proven via `congr_arg` with well-foundedness
  - Axioms proven as theorems:
    - **S1** `s1_distributivity`: left distributivity over addition
    - **S2** `s2_mul_inverse`: inverse law via `rw [mul_toReal]` + `nlinarith`
    - **S3** `s3_commutativity_with_add`: cross-structure commutativity

- **`formal/README.md`** updated — build instructions with `lake build`

#### CI/CD
- `qa-gates.yml` in StudyNinja-Eco: lean-formal job now builds Balansis formal proofs in matrix strategy alongside MagicBrain

### Infrastructure
- `.gitignore` created for Python + Lean4 build artifacts
- Remote URL migrated to SSH + XTeam-Pro organization
- `development` branch established as default for active work

---

## [0.1.0] — 2025-01-XX — Initial Release

### Added
- Initial implementation of Absolute Compensation Theory (ACT)
- Core mathematical components:
  - `AbsoluteValue` class with magnitude and direction
  - `EternalRatio` class for stable fraction representation
  - `Compensator` engine for numerical stability
- Algebraic structures:
  - `AbsoluteGroup` implementation with group theory verification
  - `EternityField` implementation with field theory verification
- Compensated arithmetic operations: `compensated_add`, `compensated_multiply`, `compensated_divide`, `compensated_power`
- Near-cancellation detection (threshold 1e-15), overflow/underflow protection
- `sequence_sum` (Kahan-compensated), `sequence_product`
- Linear algebra: `gemm.py` (compensated GEMM), `svd.py` (Golub-Kahan + QR), `qr.py` (Householder/Givens/Gram-Schmidt)
- ML optimizer: `EternalOptimizer`, `AdaptiveEternalOptimizer`, `EternalTorchOptimizer` (PyTorch subclass)
- Finance module: `finance/ledger.py` (exact cancellation accounting)
- NumPy integration: `numpy_integration.py` (vectorized ACT ops)
- Memory: `memory/arena.py` (value pooling)
- Lean4 formal specs (initial): `formal/ACT/Absolute.lean`, `Eternity.lean`, `Algebra.lean`
- Comprehensive test suite with ≥95% coverage
- Example Jupyter notebooks demonstrating core concepts
- Poetry-based dependency management
- Type safety with MyPy strict mode
- Code quality tools: Black, isort, flake8, bandit, codespell, interrogate
- Theoretical documentation:
  - `docs/theory/act_whitepaper.md` — formal specification and axiomatics
  - `docs/theory/algebraic_proofs.md` — algebraic proofs and edge case analysis
  - `docs/guide/precision_and_stability.md` — precision guide with benchmark comparisons

### Security
- No known security vulnerabilities in initial release

---

## Versioning Policy

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### API Stability Guarantees

| API Layer | Stability |
|-----------|-----------|
| Core types (`AbsoluteValue`, `EternalRatio`) | Stable — MAJOR version only |
| Algebraic structures (`AbsoluteGroup`, `EternityField`) | Stable — MAJOR version only |
| Compensated operations | Stable — MAJOR version only |
| Utility functions, integration patterns | Evolving — MINOR version |
| Lean4 formal specs | Evolving — MINOR version |
| Private methods, test utilities | No guarantees |

### Deprecation Policy

Features marked for removal will:
1. Be deprecated for at least one MINOR version with warnings
2. Have migration paths documented in the changelog
3. Be removed only in MAJOR releases

---

*This changelog helps users and developers track the evolution of Balansis and make informed decisions about upgrades and compatibility.*
