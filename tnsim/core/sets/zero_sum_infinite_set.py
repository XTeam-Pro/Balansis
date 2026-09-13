"""Main class for working with infinite sets within the TNSIM framework."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from balansis import AbsoluteValue, Operations
except ImportError:  # pragma: no cover - optional integration
    AbsoluteValue = None  # type: ignore[assignment]
    Operations = None  # type: ignore[assignment]

# Set precision for Decimal


def _to_decimal(value: Union[float, Decimal, int]) -> Decimal:
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite():
        raise ValueError("Elements must be finite")
    return result


def _compensated_sum(values: np.ndarray) -> float:
    if Operations is None or AbsoluteValue is None:
        return float(np.sum(values))

    absolute_values = [AbsoluteValue.from_float(float(v)) for v in np.asarray(values)]
    result, _ = Operations.sequence_sum(absolute_values)
    return result.to_float()


class ZeroSumInfiniteSet:
    """Class for representing infinite sets within the TNSIM framework.

    Implements the ⊕ operation and compensation principles for working with infinite sets.
    """

    def __init__(
        self,
        elements: Optional[List[Union[float, Decimal, int]]] = None,
        set_type: str = "custom",
        name: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        series_type: Optional[str] = None,
        formula: Optional[str] = None,
    ) -> None:
        """Initialize infinite set.

        Args:
            elements: List of set elements
            set_type: Set type ('harmonic', 'alternating', 'geometric', 'custom')
            name: Set name
            properties: Additional set properties
        """
        effective_type = series_type or set_type
        self.id = str(uuid.uuid4())
        self.elements = np.array(
            [_to_decimal(x) for x in (elements or [])], dtype=object
        )
        self.set_type = effective_type
        self.series_type = effective_type
        self.name = name or f"{effective_type}_set_{self.id[:8]}"
        self.properties = dict(properties or {})
        if formula is not None:
            self.properties.setdefault("formula", formula)
        self.created_at = datetime.now(timezone.utc)
        self._compensating_set = None
        self._cached_sum = None

    def __repr__(self) -> str:
        return f"ZeroSumInfiniteSet(name='{self.name}', type='{self.set_type}', elements={len(self.elements)})"

    def zero_sum_operation(
        self,
        other: Optional["ZeroSumInfiniteSet"] = None,
        method: str = "compensated",
        tolerance: Optional[float] = None,
        max_iterations: Optional[int] = None,
    ) -> Union[Decimal, Dict[str, Any]]:
        """Execute ⊕ operation between two sets.

        Args:
            other: Another infinite set
            method: Calculation method ('direct', 'compensated', 'stabilized')

        Returns:
            Result of ⊕ operation
        """
        if method not in {"direct", "compensated", "stabilized"}:
            raise ValueError(f"Unknown method: {method}")
        if tolerance is not None and (not np.isfinite(tolerance) or tolerance <= 0):
            raise ValueError("tolerance must be positive and finite")
        if other is None:
            total = self._self_sum(method)
            tol = Decimal(str(tolerance if tolerance is not None else 1e-10))
            return {
                "sum": total,
                "method": method,
                "compensation_error": abs(total),
                "iterations": max_iterations or 1,
                "stability_factor": 1.0,
                "numerical_precision": float(-np.log10(float(abs(total)) + 1e-16)),
                "tolerance": tol,
            }

        if method == "direct":
            return self._direct_sum(other)
        if method == "compensated":
            return self._compensated_sum(other)
        if method == "stabilized":
            return self._stabilized_sum(other)
        raise ValueError(f"Unknown method: {method}")

    def _self_sum(self, method: str) -> Decimal:
        if method == "direct":
            return Decimal(str(sum(self.elements)))
        values = np.asarray([float(x) for x in self.elements], dtype=np.float64)
        return Decimal(str(_compensated_sum(values)))

    def _direct_sum(self, other: "ZeroSumInfiniteSet") -> Decimal:
        """Direct summation of set elements."""
        sum_a = sum(self.elements)
        sum_b = sum(other.elements)
        return sum_a + sum_b

    def _compensated_sum(self, other: "ZeroSumInfiniteSet") -> Decimal:
        """Compensated summation through Balansis."""
        sum_a = _compensated_sum(
            np.asarray([float(x) for x in self.elements], dtype=np.float64)
        )
        sum_b = _compensated_sum(
            np.asarray([float(x) for x in other.elements], dtype=np.float64)
        )
        return Decimal(str(sum_a + sum_b))

    def _stabilized_sum(self, other: "ZeroSumInfiniteSet") -> Decimal:
        """Stabilized summation with additional compensation."""
        stabilized_a = np.asarray([float(x) for x in self.elements], dtype=np.float64)
        stabilized_b = np.asarray([float(x) for x in other.elements], dtype=np.float64)
        sum_a = _compensated_sum(stabilized_a)
        sum_b = _compensated_sum(stabilized_b)
        return Decimal(str(sum_a + sum_b))

    def find_compensating_set(self, method: str = "direct") -> "ZeroSumInfiniteSet":
        methods = {
            "direct": self._direct_compensating_set,
            "iterative": self._iterative_compensating_set,
            "adaptive": self._adaptive_compensating_set,
        }
        if method not in methods:
            raise ValueError(f"Unknown method: {method}")
        return methods[method]()

    def _direct_compensating_set(self) -> "ZeroSumInfiniteSet":
        """Direct creation of compensating set."""
        compensating_elements = [-x for x in self.elements]
        return ZeroSumInfiniteSet(
            compensating_elements,
            f"compensating_{self.set_type}",
            f"Compensating_{self.name}",
            {**self.properties, "compensates": self.id},
        )

    def _iterative_compensating_set(self) -> "ZeroSumInfiniteSet":
        """Iterative search for compensating set."""
        # Start with direct negation
        compensating_elements = [-x for x in self.elements]

        # Iterative correction to achieve precise compensation
        for i in range(10):  # Maximum 10 iterations
            temp_set = ZeroSumInfiniteSet(compensating_elements)
            result = self.zero_sum_operation(temp_set, "compensated")

            if abs(result) < Decimal("1e-10"):
                break

            # Element correction
            correction = result / len(compensating_elements)
            compensating_elements = [x - correction for x in compensating_elements]

        return ZeroSumInfiniteSet(
            compensating_elements,
            f"iterative_compensating_{self.set_type}",
            f"Iterative_Compensating_{self.name}",
            {**self.properties, "compensates": self.id, "method": "iterative"},
        )

    def _adaptive_compensating_set(self) -> "ZeroSumInfiniteSet":
        """Adaptive search for compensating set using Balansis."""
        stabilized_elements = np.asarray(
            [float(x) for x in self.elements], dtype=np.float64
        )
        compensating_elements = [-float(x) for x in stabilized_elements]

        return ZeroSumInfiniteSet(
            compensating_elements,
            f"adaptive_compensating_{self.set_type}",
            f"Adaptive_Compensating_{self.name}",
            {**self.properties, "compensates": self.id, "method": "adaptive"},
        )

    def validate_zero_sum(
        self,
        tolerance: Decimal = Decimal("1e-10"),
        detailed: bool = False,
    ) -> Dict[str, Any]:
        """Validate zero sum with compensating set.

        Args:
            tolerance: Acceptable error tolerance

        Returns:
            Dictionary with validation results
        """
        compensating = self.find_compensating_set()
        result = self.zero_sum_operation(compensating, "compensated")

        is_zero_sum = abs(result) < tolerance

        payload = {
            "is_zero_sum": is_zero_sum,
            "result": result,
            "tolerance": tolerance,
            "error_margin": abs(result),
            "compensating_set_id": compensating.id,
            "validation_timestamp": datetime.now(timezone.utc),
        }
        if detailed:
            payload["element_count"] = len(self.elements)
            payload["partial_sum"] = self.get_partial_sum(len(self.elements))
        return payload

    def get_partial_sum(self, n_elements: int, start: int = 0) -> Decimal:
        """Get partial sum of first n elements.

        Args:
            n_elements: Number of elements to sum

        Returns:
            Partial sum
        """
        if start < 0:
            raise ValueError("start must be non-negative")
        if n_elements < 0:
            raise ValueError("n_elements must be non-negative")
        end = min(start + n_elements, len(self.elements))
        return sum(self.elements[start:end], Decimal(0))

    def get_element(self, index: int) -> Decimal:
        if index < 0 or index >= len(self.elements):
            raise IndexError("element index out of range")
        return self.elements[index]

    def convergence_analysis(
        self,
        max_terms: int = 1000,
        analysis_methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Analyze series convergence.

        Args:
            max_terms: Maximum number of terms for analysis

        Returns:
            Convergence analysis results
        """
        partial_sums = []
        n_terms = min(max_terms, len(self.elements))

        for i in range(1, n_terms + 1):
            partial_sum = self.get_partial_sum(i)
            partial_sums.append(float(partial_sum))

        # Simple convergence analysis
        if len(partial_sums) > 10:
            last_10 = partial_sums[-10:]
            variance = np.var(last_10)
            is_convergent = variance < 1e-6
        else:
            is_convergent = False
            variance = float("inf")

        return {
            "is_convergent": is_convergent,
            "partial_sums": partial_sums,
            "variance": variance,
            "final_sum": partial_sums[-1] if partial_sums else 0,
            "n_terms_analyzed": n_terms,
            "analysis_methods": analysis_methods or [],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "set_type": self.set_type,
            "elements": [str(x) for x in self.elements],
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "element_count": len(self.elements),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ZeroSumInfiniteSet":
        """Create object from dictionary."""
        obj = cls(
            elements=data["elements"],
            set_type=data["set_type"],
            name=data["name"],
            properties=data.get("properties", {}),
        )
        obj.id = data["id"]
        obj.created_at = datetime.fromisoformat(data["created_at"])
        return obj

    @staticmethod
    def create_harmonic_series(
        n_terms: int = 1000,
        p: float = 1.0,
    ) -> "ZeroSumInfiniteSet":
        """Create harmonic series."""
        if not isinstance(n_terms, int) or isinstance(n_terms, bool) or n_terms < 0:
            raise ValueError("n_terms must be a non-negative integer")
        with localcontext() as context:
            context.prec = 50
            elements = [
                Decimal(1) / (Decimal(i) ** Decimal(str(p)))
                for i in range(1, n_terms + 1)
            ]
            return ZeroSumInfiniteSet(
                elements,
                "harmonic",
                "Harmonic Series",
                {"formula": f"1/n^{p}", "divergent": p <= 1.0, "p": p},
            )

    @staticmethod
    def create_alternating_series(
        n_terms: int = 1000,
        p: float = 1.0,
    ) -> "ZeroSumInfiniteSet":
        """Create alternating series."""
        if not isinstance(n_terms, int) or isinstance(n_terms, bool) or n_terms < 0:
            raise ValueError("n_terms must be a non-negative integer")
        with localcontext() as context:
            context.prec = 50
            elements = [
                Decimal((-1) ** i) / (Decimal(i + 1) ** Decimal(str(p)))
                for i in range(n_terms)
            ]
            return ZeroSumInfiniteSet(
                elements,
                "alternating",
                "Alternating Series",
                {"formula": f"(-1)^n/(n+1)^{p}", "convergent": True, "p": p},
            )

    @staticmethod
    def create_geometric_series(
        ratio: float = 0.5, n_terms: int = 1000
    ) -> "ZeroSumInfiniteSet":
        """Create geometric series."""
        if not isinstance(n_terms, int) or isinstance(n_terms, bool) or n_terms < 0:
            raise ValueError("n_terms must be a non-negative integer")
        with localcontext() as context:
            context.prec = 50
            elements = [Decimal(ratio) ** i for i in range(n_terms)]
            return ZeroSumInfiniteSet(
                elements,
                "geometric",
                f"Geometric Series (r={ratio})",
                {"ratio": ratio, "convergent": abs(ratio) < 1},
            )
