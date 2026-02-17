# Changelog

All notable changes to the Balansis project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Complete theoretical documentation for Absolute Compensation Theory (ACT)
- Formal mathematical specification in `docs/theory/act_whitepaper.md`
- Algebraic proofs and edge case analysis in `docs/theory/algebraic_proofs.md`
- Comprehensive precision and stability guide in `docs/guide/precision_and_stability.md`
- Integration patterns for NumPy, Pandas, and PyTorch
- Diagnostic tools for stability analysis and performance profiling
- Comparative benchmarks with IEEE 754, Kahan summation, and Python Decimal

### Changed
- Updated README.md with comprehensive installation instructions
- Enhanced project documentation structure
- Added academic research section to README

### Fixed
- Clarified installation methods (Poetry vs pip)
- Added proper documentation links throughout the project

## [0.5.0] - 2026-02-17

### Added
- ACT-Compensated SVD (Householder bidiagonalization + QR iteration)
- ACT-Compensated QR (Householder reflections, Givens, Modified Gram-Schmidt)
- AdaptiveEternalOptimizer (Adam-like with ACT moments, gradient clipping, warmup/cosine decay)
- Enhanced numpy integration (vectorized ACT operations: compensated_array_add, compensated_array_multiply, compensated_dot_product, compensated_outer_product, compensated_softmax)
- Lean4 formal specification (12 axioms)
- CI-integrated benchmark suite with regression tracking
- ACT Specification v1.0 document

## [0.1.0] - 2024-01-XX

### Added
- Initial implementation of Absolute Compensation Theory (ACT)
- Core mathematical components:
  - `AbsoluteValue` class with magnitude and direction
  - `EternalRatio` class for stable fraction representation
  - `Compensator` engine for numerical stability
- Algebraic structures:
  - `AbsoluteGroup` implementation with group theory verification
  - `EternityField` implementation with field theory verification
- Basic arithmetic operations with compensation logic
- Comprehensive test suite with >95% coverage
- Example Jupyter notebooks demonstrating core concepts
- Poetry-based dependency management
- Type safety with MyPy
- Code quality tools (Black, isort, flake8)

### Security
- No known security vulnerabilities in initial release

---

## Versioning Policy

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html):

- **MAJOR** version when you make incompatible API changes
- **MINOR** version when you add functionality in a backwards compatible manner  
- **PATCH** version when you make backwards compatible bug fixes

### Version Number Format: `MAJOR.MINOR.PATCH`

### Pre-release Versions
Pre-release versions may be denoted by appending a hyphen and a series of dot separated identifiers:
- `1.0.0-alpha` - Alpha release
- `1.0.0-beta` - Beta release  
- `1.0.0-rc.1` - Release candidate

### API Stability Guarantees

#### Stable APIs (MAJOR version changes only)
- Core mathematical operations (`AbsoluteValue`, `EternalRatio`)
- Public methods of algebraic structures (`AbsoluteGroup`, `EternityField`)
- Compensator public interface

#### Evolving APIs (MINOR version changes)
- Utility functions and helper methods
- Visualization and plotting tools
- Integration patterns and examples

#### Internal APIs (No stability guarantees)
- Private methods and internal implementations
- Test utilities and debugging tools
- Development and build scripts

### Deprecation Policy

1. **Deprecation Notice**: Features marked for removal will be deprecated for at least one MINOR version
2. **Warning Period**: Deprecated features will emit warnings when used
3. **Documentation**: All deprecations will be clearly documented in the changelog
4. **Migration Path**: Alternative approaches will be provided for deprecated features

### Breaking Changes

Breaking changes will only be introduced in MAJOR version releases and will be:
- Clearly documented in the changelog
- Accompanied by migration guides
- Preceded by deprecation warnings when possible

### Compatibility Matrix

| Balansis Version | Python Version | Dependencies |
|------------------|----------------|--------------|
| 0.1.x            | 3.10+          | See pyproject.toml |
| 1.0.x            | 3.10+          | TBD |

### Release Schedule

- **PATCH** releases: As needed for bug fixes
- **MINOR** releases: Monthly or bi-monthly for new features
- **MAJOR** releases: Annually or when significant API changes are required

### Support Policy

- **Current MAJOR version**: Full support with bug fixes and security updates
- **Previous MAJOR version**: Security updates only for 12 months after new MAJOR release
- **Older versions**: Community support only

---

## Contributing to the Changelog

When contributing to Balansis, please update this changelog according to these guidelines:

### Categories
Use these categories for organizing changes:
- **Added** for new features
- **Changed** for changes in existing functionality  
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerability fixes

### Format
- Keep an "Unreleased" section at the top for upcoming changes
- Add new versions in reverse chronological order
- Use ISO date format (YYYY-MM-DD) for release dates
- Link to relevant issues and pull requests when applicable
- Write clear, concise descriptions that users can understand

### Examples

```markdown
### Added
- New `compensated_multiply` method for enhanced precision multiplication (#123)
- Support for complex number operations in AbsoluteValue (#145)

### Fixed  
- Resolved numerical instability in edge case division by near-zero values (#156)
- Fixed memory leak in large-scale compensated summation (#167)

### Changed
- Improved performance of EternalRatio operations by 25% (#134)
- Updated error messages to be more descriptive (#142)

### Deprecated
- `legacy_add` method is deprecated, use `compensated_add` instead (#178)
```

---

*This changelog helps users and developers track the evolution of Balansis and make informed decisions about upgrades and compatibility.*