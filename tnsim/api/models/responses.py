"""Response models for TNSIM API."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OperationStatus(str, Enum):
    """Operation statuses."""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    CACHED = "cached"


class ConvergenceType(str, Enum):
    """Convergence types."""

    CONVERGENT = "convergent"
    DIVERGENT = "divergent"
    CONDITIONALLY_CONVERGENT = "conditionally_convergent"
    UNKNOWN = "unknown"


class InfiniteSetResponse(BaseModel):
    """Response with infinite set information."""

    id: str = Field(..., description="Unique set identifier")
    name: str = Field(..., description="Set name")
    series_type: str = Field(..., description="Series type")
    parameters: Dict[str, Any] = Field(..., description="Series parameters")
    description: Optional[str] = Field(None, description="Set description")
    created_at: datetime = Field(..., description="Creation time")
    convergence_info: Optional[Dict[str, Any]] = Field(
        None, description="Convergence information"
    )
    partial_sums: Optional[List[float]] = Field(
        None, description="Partial sums (first N elements)"
    )


class ZeroSumOperationResponse(BaseModel):
    """Zero sum operation response."""

    operation_id: str = Field(..., description="Operation identifier")
    status: OperationStatus = Field(..., description="Operation status")
    result: Optional[float] = Field(None, description="Operation result")
    method_used: str = Field(..., description="Method used")
    iterations: int = Field(..., description="Number of iterations")
    tolerance_achieved: float = Field(..., description="Achieved tolerance")
    execution_time: float = Field(..., description="Execution time in seconds")
    set_ids: List[str] = Field(..., description="Participating set identifiers")
    compensation_details: Optional[Dict[str, Any]] = Field(
        None, description="Compensation details"
    )
    cached: bool = Field(False, description="Result obtained from cache")


class CompensatingSetResponse(BaseModel):
    """Compensating set search response."""

    operation_id: str = Field(..., description="Operation identifier")
    status: OperationStatus = Field(..., description="Operation status")
    target_set_id: str = Field(..., description="Target set identifier")
    compensating_set: Optional[InfiniteSetResponse] = Field(
        None, description="Found compensating set"
    )
    compensation_quality: Optional[float] = Field(
        None, description="Compensation quality (0-1)"
    )
    method_used: str = Field(..., description="Search method used")
    search_iterations: int = Field(..., description="Number of search iterations")
    execution_time: float = Field(..., description="Execution time in seconds")
    search_details: Optional[Dict[str, Any]] = Field(None, description="Search details")


class ValidationResponse(BaseModel):
    """Zero sum validation response."""

    validation_id: str = Field(..., description="Validation identifier")
    is_valid: bool = Field(..., description="Validation result")
    sum_value: float = Field(..., description="Sum value")
    tolerance_used: float = Field(..., description="Tolerance used")
    set_ids: List[str] = Field(..., description="Validated sets")
    convergence_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Convergence analysis"
    )
    partial_sums_analysis: Optional[List[Dict[str, Any]]] = Field(
        None, description="Partial sums analysis"
    )
    execution_time: float = Field(..., description="Execution time in seconds")


class BatchOperationResponse(BaseModel):
    """Batch operations response."""

    batch_id: str = Field(..., description="Batch identifier")
    total_operations: int = Field(..., description="Total number of operations")
    successful_operations: int = Field(..., description="Successful operations")
    failed_operations: int = Field(..., description="Failed operations")
    results: List[Dict[str, Any]] = Field(..., description="Operation results")
    execution_time: float = Field(..., description="Total execution time")
    parallel_execution: bool = Field(..., description="Parallel execution")


class ConvergenceAnalysisResponse(BaseModel):
    """Convergence analysis response."""

    set_id: str = Field(..., description="Set identifier")
    convergence_type: ConvergenceType = Field(..., description="Convergence type")
    convergence_rate: Optional[float] = Field(None, description="Convergence rate")
    analysis_methods: Dict[str, Dict[str, Any]] = Field(
        ..., description="Results of various analysis methods"
    )
    partial_sums: List[float] = Field(..., description="Partial sums")
    convergence_plot_data: Optional[Dict[str, List[float]]] = Field(
        None, description="Data for plotting graphs"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations"
    )


class ErrorResponse(BaseModel):
    """Error response."""

    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp",
    )
    request_id: Optional[str] = Field(None, description="Request identifier")


class HealthResponse(BaseModel):
    """Service health check response."""

    status: str = Field(..., description="Service status")
    database: str = Field(..., description="Database status")
    cache: Optional[str] = Field(None, description="Cache status")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Uptime in seconds")


class CacheStatsResponse(BaseModel):
    """Cache statistics response."""

    total_operations: int = Field(..., description="Total number of operations")
    cache_hits: int = Field(..., description="Cache hits")
    cache_misses: int = Field(..., description="Cache misses")
    hit_rate: float = Field(..., description="Hit rate percentage")
    cache_size: int = Field(..., description="Cache size")
    memory_usage: float = Field(..., description="Memory usage in MB")
