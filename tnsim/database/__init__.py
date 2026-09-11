"""Database module for TNSIM."""

from .config import DatabaseConfig, EnvironmentConfig, db_config, get_config
from .repository import DatabaseConnection, InfiniteSetRepository

__all__ = [
    "DatabaseConfig",
    "db_config",
    "get_config",
    "EnvironmentConfig",
    "DatabaseConnection",
    "InfiniteSetRepository",
]

from .. import __version__

__author__ = "TNSIM Team"
__description__ = "Database layer for Zero Sum Theory of Infinite Sets"
