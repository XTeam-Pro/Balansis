"""TNSIM integrations with external libraries."""

from .balansis_integration import BalansisCompensator, ZeroSumAttention

__all__ = ["ZeroSumAttention", "BalansisCompensator"]

from .. import __version__

__author__ = "TNSIM Team"
__description__ = "External library integrations for Zero Sum Theory of Infinite Sets"
