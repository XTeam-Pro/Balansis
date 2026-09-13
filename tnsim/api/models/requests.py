"""Request models for TNSIM API."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class SeriesType(str, Enum):
    """Series types for creating infinite sets."""

    HARMONIC = "harmonic"
    ALTERNATING = "alternating"
    GEOMETRIC = "geometric"
    CUSTOM = "custom"


class OperationMethod(str, Enum):
    """Operation execution methods."""

    DIRECT = "direct"
    COMPENSATED = "compensated"
    STABILIZED = "stabilized"
    ITERATIVE = "iterative"
    ADAPTIVE = "adaptive"


class CreateInfiniteSetRequest(BaseModel):
    """Request to create an infinite set."""

    name: str = Field(..., description="Set name")
    series_type: SeriesType = Field(..., description="Series type")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Series parameters"
    )
    description: Optional[str] = Field(None, description="Set description")

    @field_validator("parameters")
    def validate_parameters(cls, v, values):
        """Validate parameters based on series type."""
        series_type = values.data.get("series_type")

        if series_type == SeriesType.HARMONIC:
            if "p" not in v:
                v["p"] = 1.0  # Default to regular harmonic series
        elif series_type == SeriesType.GEOMETRIC:
            if "ratio" not in v:
                raise ValueError("Geometric series requires 'ratio' parameter")
        elif series_type == SeriesType.CUSTOM:
            if "elements" not in v:
                raise ValueError("Custom series requires 'elements' parameter")

        return v


class ZeroSumOperationRequest(BaseModel):
    """Request to perform zero sum operation."""

    set_ids: List[str] = Field(..., description="Set identifiers for operation")
    method: OperationMethod = Field(
        OperationMethod.COMPENSATED, description="Execution method"
    )
    tolerance: float = Field(1e-10, description="Acceptable tolerance")
    max_iterations: int = Field(1000, description="Maximum number of iterations")
    use_cache: bool = Field(True, description="Use caching")

    @field_validator("set_ids")
    def validate_set_ids(cls, v):
        """Validate list of set identifiers."""
        if len(v) < 2:
            raise ValueError("Zero sum operation requires at least 2 sets")
        return v

    @field_validator("tolerance")
    def validate_tolerance(cls, v):
        """Validate tolerance value."""
        if v <= 0:
            raise ValueError("Tolerance must be positive")
        return v


class FindCompensatingSetRequest(BaseModel):
    """Request to find compensating set."""

    target_set_id: str = Field(..., description="Target set identifier")
    method: OperationMethod = Field(
        OperationMethod.ADAPTIVE, description="Search method"
    )
    tolerance: float = Field(1e-10, description="Acceptable tolerance")
    max_iterations: int = Field(1000, description="Maximum number of iterations")
    search_space: Optional[Dict[str, Any]] = Field(None, description="Search space")


class ValidateZeroSumRequest(BaseModel):
    """Request to validate zero sum."""

    set_ids: List[str] = Field(..., description="Set identifiers for validation")
    tolerance: float = Field(1e-10, description="Acceptable tolerance")
    validation_method: str = Field("standard", description="Validation method")
    include_details: bool = Field(False, description="Include detailed information")

    @field_validator("set_ids")
    def validate_set_ids(cls, v):
        """Validate list of identifiers."""
        if len(v) < 2:
            raise ValueError("At least 2 sets are required for validation")
        return v


class BatchOperationRequest(BaseModel):
    """Request to perform batch operation."""

    operations: List[Dict[str, Any]] = Field(
        ..., description="List of operations to execute"
    )
    parallel: bool = Field(True, description="Execute operations in parallel")
    max_workers: Optional[int] = Field(None, description="Maximum number of workers")
    timeout: Optional[float] = Field(None, description="Execution timeout in seconds")
    stop_on_error: bool = Field(False, description="Stop on first error")

    @field_validator("operations")
    def validate_operations(cls, v):
        """Validate list of operations."""
        if not v:
            raise ValueError("Operations list cannot be empty")

        for i, op in enumerate(v):
            if "type" not in op:
                raise ValueError(f"Operation {i} must contain 'type' field")
            if op["type"] not in [
                "create",
                "zero_sum",
                "find_compensating",
                "validate",
            ]:
                raise ValueError(f"Unknown operation type: {op['type']}")

        return v


class ConvergenceAnalysisRequest(BaseModel):
    """Request for convergence analysis."""

    set_id: str = Field(..., description="Set identifier")
    max_terms: int = Field(10000, description="Maximum number of terms for analysis")
    analysis_methods: List[str] = Field(
        default=["ratio_test", "root_test", "integral_test"],
        description="Convergence analysis methods",
    )

    @field_validator("max_terms")
    def validate_max_terms(cls, v):
        if v <= 0:
            raise ValueError("Number of terms must be positive")
        if v > 100000:
            raise ValueError("Too many terms (maximum 100000)")
        return v
