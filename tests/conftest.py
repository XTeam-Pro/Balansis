"""Pytest configuration and shared fixtures for Balansis tests.

This module provides common test fixtures, configuration, and utilities
for the comprehensive test suite of the Balansis mathematical library.
"""

from decimal import Decimal, getcontext
from typing import Any, Dict, Generator, List

import numpy as np
import pytest

from balansis.core.absolute import AbsoluteValue
from balansis.core.eternity import EternalRatio
from balansis.logic.compensator import CompensationStrategy, Compensator

# Conditional import to avoid matplotlib issues in testing
try:
    from balansis.utils.plot import PlotConfig, PlotUtils

    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    PlotConfig = None
    PlotUtils = None
from balansis import ACT_EPSILON

# Set high precision for Decimal calculations in tests
getcontext().prec = 50


@pytest.fixture(scope="session")
def act_epsilon() -> float:
    """Provide ACT epsilon for numerical comparisons."""
    return ACT_EPSILON


@pytest.fixture
def sample_absolute_values() -> List[AbsoluteValue]:
    """Provide sample AbsoluteValue objects for testing."""
    return [
        AbsoluteValue(magnitude=1.0, direction=1.0),
        AbsoluteValue(magnitude=2.0, direction=-1.0),
        AbsoluteValue(magnitude=3.0, direction=0.5),
        AbsoluteValue(magnitude=0.5, direction=-0.5),
        AbsoluteValue.absolute(),
        AbsoluteValue(magnitude=0.0, direction=1),
    ]


@pytest.fixture
def sample_eternal_ratios() -> List[EternalRatio]:
    """Provide sample EternalRatio objects for testing."""
    return [
        EternalRatio(
            numerator=AbsoluteValue(magnitude=6.0, direction=1.0),
            denominator=AbsoluteValue(magnitude=3.0, direction=1.0),
        ),
        EternalRatio(
            numerator=AbsoluteValue(magnitude=4.0, direction=-1.0),
            denominator=AbsoluteValue(magnitude=2.0, direction=1.0),
        ),
        EternalRatio.unity(),
        EternalRatio(
            numerator=AbsoluteValue.absolute(),
            denominator=AbsoluteValue(magnitude=1.0, direction=1.0),
        ),
    ]


@pytest.fixture
def compensator() -> Compensator:
    """Provide a Compensator instance for testing."""
    strategy = CompensationStrategy(
        stability_threshold=1e-10,
        overflow_threshold=1e10,
        underflow_threshold=1e-10,
        singularity_threshold=1e-15,
        balance_threshold=0.1,
        convergence_threshold=1e-8,
        stability_factor=0.1,
        overflow_factor=1e-5,
        underflow_factor=1e5,
        balance_factor=0.5,
    )
    return Compensator(strategy)


@pytest.fixture
def plot_config():
    """Provide a PlotConfig instance for testing."""
    if not PLOT_AVAILABLE:
        pytest.skip("Plot utilities not available")
    return PlotConfig(width=800, height=600, dpi=100, grid=True, legend=True)


@pytest.fixture
def plot_utils(plot_config):
    """Provide a PlotUtils instance for testing."""
    if not PLOT_AVAILABLE:
        pytest.skip("Plot utilities not available")
    return PlotUtils(plot_config)


@pytest.fixture
def large_values() -> List[AbsoluteValue]:
    """Provide large magnitude AbsoluteValue objects for testing."""
    return [
        AbsoluteValue(magnitude=1e10, direction=1.0),
        AbsoluteValue(magnitude=1e15, direction=-1.0),
        AbsoluteValue(magnitude=1e20, direction=0.5),
    ]


@pytest.fixture
def small_values() -> List[AbsoluteValue]:
    """Provide small magnitude AbsoluteValue objects for testing."""
    return [
        AbsoluteValue(magnitude=1e-10, direction=1.0),
        AbsoluteValue(magnitude=1e-15, direction=-1.0),
        AbsoluteValue(magnitude=1e-20, direction=0.5),
    ]


@pytest.fixture
def precision_values() -> List[AbsoluteValue]:
    """Provide high-precision AbsoluteValue objects for testing."""
    return [
        AbsoluteValue(magnitude=1.0000000000001, direction=1.0),
        AbsoluteValue(magnitude=0.9999999999999, direction=1.0),
        AbsoluteValue(magnitude=1.0 + ACT_EPSILON / 2, direction=1.0),
        AbsoluteValue(magnitude=1.0 - ACT_EPSILON / 2, direction=1.0),
    ]


@pytest.fixture
def mixed_direction_values() -> List[AbsoluteValue]:
    """Provide AbsoluteValue objects with various directions for testing."""
    directions = [1.0, -1.0, 0.5, -0.5, 0.0, 0.1, -0.9]
    return [AbsoluteValue(magnitude=1.0, direction=d) for d in directions]


@pytest.fixture
def unity_ratios() -> List[EternalRatio]:
    """Provide EternalRatio objects close to unity for testing."""
    return [
        EternalRatio.unity(),
        EternalRatio(
            numerator=AbsoluteValue(magnitude=1.0 + ACT_EPSILON / 2, direction=1.0),
            denominator=AbsoluteValue(magnitude=1.0, direction=1.0),
        ),
        EternalRatio(
            numerator=AbsoluteValue(magnitude=1.0, direction=1.0),
            denominator=AbsoluteValue(magnitude=1.0 + ACT_EPSILON / 2, direction=1.0),
        ),
    ]


@pytest.fixture
def extreme_ratios() -> List[EternalRatio]:
    """Provide EternalRatio objects with extreme values for testing."""
    return [
        EternalRatio(
            numerator=AbsoluteValue(magnitude=1e10, direction=1.0),
            denominator=AbsoluteValue(magnitude=1e-10, direction=1.0),
        ),
        EternalRatio(
            numerator=AbsoluteValue(magnitude=1e-10, direction=1.0),
            denominator=AbsoluteValue(magnitude=1e10, direction=1.0),
        ),
        EternalRatio(
            numerator=AbsoluteValue.absolute(),
            denominator=AbsoluteValue(magnitude=1e-20, direction=1.0),
        ),
    ]


@pytest.fixture
def arithmetic_sequences() -> Dict[str, List[AbsoluteValue]]:
    """Provide arithmetic sequences for testing."""
    return {
        "increasing": [
            AbsoluteValue(magnitude=float(i), direction=1.0) for i in range(1, 6)
        ],
        "decreasing": [
            AbsoluteValue(magnitude=float(6 - i), direction=1.0) for i in range(1, 6)
        ],
        "alternating": [
            AbsoluteValue(magnitude=1.0, direction=(-1.0) ** i) for i in range(5)
        ],
        "mixed": [
            AbsoluteValue(magnitude=1.0, direction=1.0),
            AbsoluteValue(magnitude=2.0, direction=-1.0),
            AbsoluteValue.absolute(),
            AbsoluteValue(magnitude=3.0, direction=0.5),
        ],
    }


@pytest.fixture
def compensation_test_cases() -> List[Dict[str, Any]]:
    """Provide test cases for compensation testing."""
    return [
        {
            "name": "overflow_case",
            "value": AbsoluteValue(magnitude=1e15, direction=1.0),
            "expected_compensation": True,
            "compensation_type": "overflow",
        },
        {
            "name": "underflow_case",
            "value": AbsoluteValue(magnitude=1e-15, direction=1.0),
            "expected_compensation": True,
            "compensation_type": "underflow",
        },
        {
            "name": "stable_case",
            "value": AbsoluteValue(magnitude=1.0, direction=1.0),
            "expected_compensation": False,
            "compensation_type": None,
        },
        {
            "name": "absolute_case",
            "value": AbsoluteValue.absolute(),
            "expected_compensation": True,
            "compensation_type": "singularity",
        },
    ]


@pytest.fixture
def mathematical_constants() -> Dict[str, AbsoluteValue]:
    """Provide mathematical constants as AbsoluteValue objects."""
    return {
        "pi": AbsoluteValue(magnitude=np.pi, direction=1.0),
        "e": AbsoluteValue(magnitude=np.e, direction=1.0),
        "golden_ratio": AbsoluteValue(magnitude=(1 + np.sqrt(5)) / 2, direction=1.0),
        "sqrt_2": AbsoluteValue(magnitude=np.sqrt(2), direction=1.0),
        "euler_gamma": AbsoluteValue(magnitude=0.5772156649015329, direction=1.0),
    }


@pytest.fixture
def performance_test_sizes() -> List[int]:
    """Provide sizes for performance testing."""
    return [10, 100, 1000, 5000]


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m 'not slow'')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line(
        "markers", "plotting: marks tests that require plotting libraries"
    )
    config.addinivalue_line(
        "markers", "mathematical: marks tests for mathematical properties"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark slow tests
        if "performance" in item.name or "large" in item.name:
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if "integration" in item.name or "end_to_end" in item.name:
            item.add_marker(pytest.mark.integration)

        # Mark plotting tests
        if "plot" in item.name or "visualization" in item.name:
            item.add_marker(pytest.mark.plotting)

        # Mark mathematical property tests
        if (
            "axiom" in item.name
            or "property" in item.name
            or "mathematical" in item.name
        ):
            item.add_marker(pytest.mark.mathematical)


# Custom assertions for ACT testing
def assert_act_equal(
    actual: AbsoluteValue, expected: AbsoluteValue, epsilon: float = ACT_EPSILON
):
    """Assert that two AbsoluteValue objects are equal within ACT epsilon."""
    assert (
        abs(actual.magnitude - expected.magnitude) < epsilon
    ), f"Magnitudes differ: {actual.magnitude} != {expected.magnitude}"
    assert (
        abs(actual.direction - expected.direction) < epsilon
    ), f"Directions differ: {actual.direction} != {expected.direction}"


def assert_ratio_equal(
    actual: EternalRatio, expected: EternalRatio, epsilon: float = ACT_EPSILON
):
    """Assert that two EternalRatio objects are equal within ACT epsilon."""
    assert (
        abs(actual.numerical_value - expected.numerical_value) < epsilon
    ), f"Numerical values differ: {actual.numerical_value} != {expected.numerical_value}"
    assert (
        abs(actual.signed_value - expected.signed_value) < epsilon
    ), f"Signed values differ: {actual.signed_value} != {expected.signed_value}"


def assert_compensation_preserves_act(
    original: AbsoluteValue, compensated: AbsoluteValue
):
    """Assert that compensation preserves ACT axioms."""
    # Compensation axiom: compensated value should be more stable
    # This is a simplified check - in practice, stability is context-dependent
    assert compensated.magnitude > 0, "Compensated value should have positive magnitude"
    assert (
        abs(compensated.direction) <= 1.0
    ), "Compensated direction should be normalized"

    # Eternity axiom: structural relationships should be preserved
    if not original.is_absolute and not compensated.is_absolute:
        ratio = EternalRatio(numerator=compensated, denominator=original)
        assert ratio.is_stable, "Compensation ratio should be stable"


# Test data generators
def generate_random_absolute_values(count: int, seed: int = 42) -> List[AbsoluteValue]:
    """Generate random AbsoluteValue objects for testing."""
    np.random.seed(seed)
    values = []

    for _ in range(count):
        magnitude = np.random.uniform(1e-5, 1e5)
        direction = np.random.uniform(-1.0, 1.0)
        values.append(AbsoluteValue(magnitude=magnitude, direction=direction))

    return values


def generate_random_eternal_ratios(count: int, seed: int = 42) -> List[EternalRatio]:
    """Generate random EternalRatio objects for testing."""
    np.random.seed(seed)
    ratios = []

    for _ in range(count):
        num_mag = np.random.uniform(1e-3, 1e3)
        num_dir = np.random.uniform(-1.0, 1.0)
        den_mag = np.random.uniform(1e-3, 1e3)
        den_dir = np.random.uniform(-1.0, 1.0)

        numerator = AbsoluteValue(magnitude=num_mag, direction=num_dir)
        denominator = AbsoluteValue(magnitude=den_mag, direction=den_dir)

        ratios.append(EternalRatio(numerator=numerator, denominator=denominator))

    return ratios


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up any test files created during testing."""
    import glob
    import os

    yield  # Run the test

    # Clean up test output files
    test_files = (
        glob.glob("test_*.png")
        + glob.glob("test_*.csv")
        + glob.glob("test_*.json")
        + glob.glob("test_*.xlsx")
    )

    for file in test_files:
        try:
            os.remove(file)
        except OSError:
            pass  # File might not exist or be in use
