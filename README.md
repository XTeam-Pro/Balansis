# Balansis

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Poetry](https://img.shields.io/badge/dependency%20manager-poetry-blue.svg)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checker: MyPy](https://img.shields.io/badge/type%20checker-mypy-blue.svg)](http://mypy-lang.org/)

**A revolutionary Python mathematical library implementing Absolute Compensation Theory (ACT)**

Balansis transforms computational mathematics by replacing traditional zero and infinity with mathematically stable "Absolute" and "Eternity" concepts, eliminating numerical instabilities and singularities that plague modern computing.

## 🚀 What is Absolute Compensation Theory?

Absolute Compensation Theory (ACT) is a groundbreaking mathematical framework that addresses fundamental computational instabilities:

- **Replaces Zero** with "Absolute" - a stable mathematical entity that prevents division by zero
- **Replaces Infinity** with "Eternity" - a bounded concept that eliminates overflow conditions
- **Introduces Compensation Logic** - automatic stability mechanisms for edge cases
- **Ensures Mathematical Consistency** - maintains algebraic properties while enhancing stability

## ✨ Key Features

### Core Mathematical Components
- **`AbsoluteValue`** - Enhanced numerical values with magnitude and direction
- **`EternalRatio`** - Stable ratio representations replacing traditional fractions
- **`Compensator`** - Intelligent stability engine for mathematical operations

### Advanced Algebraic Structures
- **`AbsoluteGroup`** - Group theory implementation for AbsoluteValue objects
- **`EternityField`** - Field theory implementation for EternalRatio objects
- **Axiom Verification** - Automatic testing of mathematical properties

### Practical Benefits
- 🛡️ **Eliminates Division by Zero** - No more runtime exceptions
- 📈 **Prevents Numerical Overflow** - Stable computations at scale
- 🔬 **Enhanced Precision** - Compensated arithmetic algorithms
- 🧮 **Mathematical Consistency** - Preserves algebraic properties
- 🔧 **Easy Integration** - Drop-in replacement for standard operations

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- Poetry (recommended) or pip

### Using Poetry (Recommended)
```bash
# Clone the repository
git clone https://github.com/your-username/balansis.git
cd balansis

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Using pip
```bash
pip install balansis              # core (pydantic + numpy)
pip install balansis[plot]        # + matplotlib, plotly
pip install balansis[notebook]    # + jupyter, ipykernel
pip install balansis[all]         # all optional dependencies
```

### Development Installation
```bash
# Clone the repository
git clone https://github.com/your-username/balansis.git
cd balansis

# Install with pip in development mode
pip install -e .

# Or install with all development dependencies
pip install -e ".[dev]"
```

## 🚀 Quick Start

### Basic Usage

```python
from balansis.core import AbsoluteValue, EternalRatio
from balansis.algebra import AbsoluteGroup, EternityField

# Create AbsoluteValue objects
a = AbsoluteValue(magnitude=5.0, direction=1)
b = AbsoluteValue(magnitude=3.0, direction=-1)

# Safe arithmetic operations
result = a + b  # Compensated addition
print(f"Result: {result}")  # No division by zero possible

# Create the special 'Absolute' value
absolute = AbsoluteValue.absolute()
print(f"Absolute: {absolute}")  # Replaces traditional zero

# EternalRatio for stable fractions
ratio = EternalRatio(numerator=a, denominator=b)
print(f"Stable ratio: {ratio}")  # No infinity issues
```

### Advanced Mathematical Structures

```python
# Work with algebraic groups
group = AbsoluteGroup()
members = [AbsoluteValue(magnitude=i, direction=1) for i in range(1, 4)]

# Verify group properties
print(f"Associativity: {group.verify_associativity(members)}")
print(f"Identity exists: {group.has_identity()}")

# Field operations with EternalRatio
field = EternityField()
ratios = [EternalRatio(AbsoluteValue(i), AbsoluteValue(j)) 
          for i, j in [(1, 2), (3, 4), (5, 6)]]

# Test field axioms
print(f"Distributivity: {field.verify_distributivity(ratios)}")
```

### Compensated Arithmetic

```python
from balansis.logic import Compensator

# Initialize compensator for enhanced stability
compensator = Compensator(precision_threshold=1e-15)

# Perform compensated operations
values = [AbsoluteValue(0.1), AbsoluteValue(0.2), AbsoluteValue(0.3)]
stable_sum = compensator.compensated_sum(values)

print(f"Stable sum: {stable_sum}")  # Higher precision than standard addition
```

## 🏗️ Project Structure

```
balansis/
├── balansis/
│   ├── core/              # Core mathematical components
│   │   ├── absolute_value.py
│   │   ├── eternal_ratio.py
│   │   └── operations.py
│   ├── algebra/           # Algebraic structures
│   │   ├── absolute_group.py
│   │   └── eternity_field.py
│   ├── logic/             # Compensation logic
│   │   └── compensator.py
│   └── utils/             # Utilities and visualization
│       └── plotting.py
├── tests/                 # Comprehensive test suite
├── examples/              # Jupyter notebook examples
│   ├── 01_introduction_to_act.ipynb
│   ├── 02_core_operations.ipynb
│   └── 03_algebraic_structures_and_applications.ipynb
├── docs/                  # Documentation
└── pyproject.toml         # Poetry configuration
```

## 📚 Documentation & Examples

### Complete Documentation
- **[ACT Whitepaper](docs/theory/act_whitepaper.md)** - Formal mathematical specification
- **[Algebraic Proofs](docs/theory/algebraic_proofs.md)** - Theoretical foundations and proofs
- **[Precision & Stability Guide](docs/guide/precision_and_stability.md)** - Practical recommendations

### Interactive Examples
Explore the `examples/` directory for comprehensive Jupyter notebooks:

1. **Introduction to ACT** - Basic concepts and theory
2. **Core Operations** - Practical usage patterns
3. **Algebraic Structures** - Advanced mathematical applications

### Key Concepts

#### AbsoluteValue
- **Magnitude**: The numerical value (always positive)
- **Direction**: Sign indicator (+1 or -1)
- **Special States**: 'Absolute' replaces zero, 'Infinite' handles large values

#### EternalRatio
- **Structural Ratios**: Stable fraction representation
- **Compensation**: Automatic handling of edge cases
- **Field Properties**: Full algebraic field implementation

#### Compensator Engine
- **Precision Control**: Configurable stability thresholds
- **Error Compensation**: Automatic correction of numerical errors
- **Algorithm Selection**: Optimal methods for different scenarios

## 🧪 Development & Testing

### Running Tests
```bash
# Run all tests with coverage
poetry run pytest --cov=balansis --cov-report=html

# Run specific test categories
poetry run pytest tests/core/
poetry run pytest tests/algebra/
```

### Code Quality
```bash
# Type checking
poetry run mypy balansis/

# Code formatting
poetry run black balansis/ tests/

# Import sorting
poetry run isort balansis/ tests/

# Linting
poetry run flake8 balansis/
```

### Development Setup
```bash
# Install development dependencies
poetry install --with dev

# Install pre-commit hooks
poetry run pre-commit install
```

## 🎯 Applications

Balansis is particularly valuable in:

- **🤖 AI/Machine Learning** - Stable neural network training, robust optimization
- **🔐 Cryptography** - Secure mathematical operations, key generation
- **🌐 Distributed Systems** - Consensus algorithms, fault-tolerant computations
- **⚛️ Physics Simulations** - Quantum mechanics, particle physics modeling
- **💰 Financial Computing** - Risk analysis, derivative pricing, portfolio optimization
- **📊 Scientific Computing** - Numerical analysis, differential equations

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive tests for new features
- Update documentation for API changes
- Ensure all tests pass and coverage remains >95%

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the need for numerical stability in modern computing
- Built with modern Python best practices and type safety
- Designed for both research and production environments

## 📞 Support

- **Documentation**: [docs/](docs/) - Complete theoretical and practical guides
- **Issues**: [GitHub Issues](https://github.com/your-username/balansis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/balansis/discussions)

## 🔬 Research & Academic Use

Balansis implements novel mathematical concepts suitable for academic research:

- **Formal ACT Specification**: Complete mathematical framework with proofs
- **Comparative Analysis**: Benchmarks against traditional methods (IEEE 754, Kahan, Decimal)
- **Stability Guarantees**: Theoretical error bounds and convergence properties
- **Reproducible Results**: Deterministic algorithms for scientific computing

For academic citations and research collaboration, please refer to our [theoretical documentation](docs/theory/).

---

**Balansis** - *Bringing mathematical stability to computational excellence*