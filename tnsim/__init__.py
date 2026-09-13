"""Finite Decimal series, compensation, caching and a FastAPI service."""

# Package version
__version__ = "1.1.0"
__author__ = "Andrey Tikhonov"
__email__ = "andrew@xteam.pro"
__description__ = "Theory of Zero-Sum Infinite Sets"
__url__ = "https://github.com/StudyLabPro/Balansis"
__license__ = "AGPL-3.0 / Commercial via parent Balansis repository"

# Import main classes
try:
    from .core import (
        ParallelTNSIM,
        TNSIMCache,
        ZeroSumInfiniteSet,
        cached_operation,
        get_global_cache,
        get_global_parallel_processor,
    )
except ImportError as e:
    import warnings

    warnings.warn(f"Failed to import main classes: {e}")
    ZeroSumInfiniteSet = None
    TNSIMCache = None
    ParallelTNSIM = None
    cached_operation = None
    get_global_cache = None
    get_global_parallel_processor = None

# Import integrations
try:
    from .integrations import BalansisCompensator, ZeroSumAttention
except ImportError as e:
    import warnings

    warnings.warn(f"Failed to import integrations: {e}")
    ZeroSumAttention = None
    BalansisCompensator = None

# Import database configuration
try:
    from .database import DatabaseConfig, db_config, get_config
except ImportError as e:
    import warnings

    warnings.warn(f"Failed to import database configuration: {e}")
    DatabaseConfig = None
    db_config = None
    get_config = None

# Exported symbols
__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "__url__",
    "__license__",
    # Main classes
    "ZeroSumInfiniteSet",
    "TNSIMCache",
    "ParallelTNSIM",
    "cached_operation",
    "get_global_cache",
    "get_global_parallel_processor",
    # Integrations
    "ZeroSumAttention",
    "BalansisCompensator",
    # Database
    "DatabaseConfig",
    "db_config",
    "get_config",
    # Utilities
    "create_harmonic_series",
    "create_alternating_series",
    "create_geometric_series",
    "run_server",
    "get_version_info",
]


# Utility functions
def create_harmonic_series(n_terms: int = 1000) -> "ZeroSumInfiniteSet":
    """Create a harmonic series.

    Args:
        n_terms: Number of elements to generate

    Returns:
        ZeroSumInfiniteSet: Harmonic series object
    """
    if ZeroSumInfiniteSet is None:
        raise ImportError("ZeroSumInfiniteSet is not available")
    return ZeroSumInfiniteSet.create_harmonic_series(n_terms)


def create_alternating_series(n_terms: int = 1000) -> "ZeroSumInfiniteSet":
    """Create an alternating series.

    Args:
        n_terms: Number of elements to generate

    Returns:
        ZeroSumInfiniteSet: Alternating series object
    """
    if ZeroSumInfiniteSet is None:
        raise ImportError("ZeroSumInfiniteSet is not available")
    return ZeroSumInfiniteSet.create_alternating_series(n_terms)


def create_geometric_series(
    ratio: float = 0.5, n_terms: int = 1000
) -> "ZeroSumInfiniteSet":
    """Create a geometric series.

    Args:
        ratio: Common ratio of the progression
        n_terms: Number of elements to generate

    Returns:
        ZeroSumInfiniteSet: Geometric series object
    """
    if ZeroSumInfiniteSet is None:
        raise ImportError("ZeroSumInfiniteSet is not available")
    return ZeroSumInfiniteSet.create_geometric_series(ratio, n_terms)


def run_server(host: str = "127.0.0.1", port: int = 8000, **kwargs):
    """Start FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to
        **kwargs: Additional parameters for uvicorn
    """
    try:
        import uvicorn

        from .api.main import app

        uvicorn.run(app, host=host, port=port, **kwargs)
    except ImportError:
        raise ImportError("uvicorn and FastAPI are required to run the server")


def get_version_info() -> dict:
    """Get version and dependency information.

    Returns:
        dict: Dictionary with version information
    """
    import platform
    import sys

    info = {
        "tnsim_version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
    }

    # Check availability of main dependencies
    dependencies = {
        "numpy": None,
        "scipy": None,
        "torch": None,
        "fastapi": None,
        "asyncpg": None,
    }

    for dep in dependencies:
        try:
            module = __import__(dep)
            dependencies[dep] = getattr(module, "__version__", "unknown")
        except ImportError:
            dependencies[dep] = "not installed"

    info["dependencies"] = dependencies

    # Check availability of TNSIM components
    components = {
        "ZeroSumInfiniteSet": ZeroSumInfiniteSet is not None,
        "TNSIMCache": TNSIMCache is not None,
        "ParallelTNSIM": ParallelTNSIM is not None,
        "ZeroSumAttention": ZeroSumAttention is not None,
        "BalansisCompensator": BalansisCompensator is not None,
        "DatabaseConfig": DatabaseConfig is not None,
    }

    info["components"] = components

    return info


# Logging setup
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Python compatibility check
import sys

if sys.version_info < (3, 10):
    raise RuntimeError(
        f"TNSIM requires Python 3.10 or higher. "
        f"Current version: {sys.version_info.major}.{sys.version_info.minor}"
    )
