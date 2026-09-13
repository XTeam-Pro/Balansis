"""Compatibility imports for the canonical group implementation.

Legacy import paths remain available. The implementation lives in
``absolute_group``; these exploratory helpers do not prove group axioms.
"""

from .absolute_group import (
    AbsoluteGroup,
    AdditiveOperation,
    GroupElement,
    GroupOperation,
    MultiplicativeOperation,
)

__all__ = [
    "AbsoluteGroup",
    "AdditiveOperation",
    "GroupElement",
    "GroupOperation",
    "MultiplicativeOperation",
]
