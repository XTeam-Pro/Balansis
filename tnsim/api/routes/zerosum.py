"""Routes for zero-sum operations on infinite sets."""

import logging
from typing import Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from ...core import get_global_cache
from ..models.requests import (
    BatchOperationRequest,
    ConvergenceAnalysisRequest,
    CreateInfiniteSetRequest,
    FindCompensatingSetRequest,
    ValidateZeroSumRequest,
    ZeroSumOperationRequest,
)
from ..models.responses import (
    BatchOperationResponse,
    CacheStatsResponse,
    CompensatingSetResponse,
    ConvergenceAnalysisResponse,
    InfiniteSetResponse,
    ValidationResponse,
    ZeroSumOperationResponse,
)
from ..services.zerosum_service import ZeroSumService

logger = logging.getLogger(__name__)
router = APIRouter()


# Dependency for getting service
def get_zerosum_service() -> ZeroSumService:
    """Get TNSIM service instance."""
    return ZeroSumService()


@router.post("/sets", response_model=InfiniteSetResponse)
async def create_infinite_set(
    request: CreateInfiniteSetRequest,
    service: ZeroSumService = Depends(get_zerosum_service),
) -> InfiniteSetResponse:
    """Create new infinite set."""
    try:
        logger.info(f"Creating set: {request.name}, type: {request.series_type}")

        result = await service.create_infinite_set(
            name=request.name,
            series_type=request.series_type,
            parameters=request.parameters,
            description=request.description,
        )

        logger.info(f"Set created with ID: {result.id}")
        return result

    except ValueError as e:
        logger.error(f"Validation error when creating set: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error when creating set: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sets/{set_id}", response_model=InfiniteSetResponse)
async def get_infinite_set(
    set_id: str, service: ZeroSumService = Depends(get_zerosum_service)
) -> InfiniteSetResponse:
    """Get information about infinite set."""
    try:
        result = await service.get_infinite_set(set_id)
        if not result:
            raise HTTPException(status_code=404, detail="Set not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting set {set_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sets", response_model=List[InfiniteSetResponse])
async def list_infinite_sets(
    limit: int = 100,
    offset: int = 0,
    service: ZeroSumService = Depends(get_zerosum_service),
) -> List[InfiniteSetResponse]:
    """Get list of infinite sets."""
    try:
        return await service.list_infinite_sets(limit=limit, offset=offset)
    except Exception as e:
        logger.error(f"Error getting list of sets: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/operations/zero-sum", response_model=ZeroSumOperationResponse)
async def perform_zero_sum_operation(
    request: ZeroSumOperationRequest,
    background_tasks: BackgroundTasks,
    service: ZeroSumService = Depends(get_zerosum_service),
) -> ZeroSumOperationResponse:
    """Perform zero-sum operation."""
    try:
        logger.info(f"Performing zero-sum operation for sets: {request.set_ids}")

        result = await service.perform_zero_sum_operation(
            set_ids=request.set_ids,
            method=request.method,
            tolerance=request.tolerance,
            max_iterations=request.max_iterations,
            use_cache=request.use_cache,
        )

        # Log result in background task
        background_tasks.add_task(
            service.log_operation,
            operation_type="zero_sum",
            operation_id=result.operation_id,
            result=result.model_dump(mode="json"),
        )

        logger.info(
            f"Operation completed: {result.operation_id}, status: {result.status}"
        )
        return result

    except ValueError as e:
        logger.error(f"Validation error in zero-sum operation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in zero-sum operation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/operations/find-compensating", response_model=CompensatingSetResponse)
async def find_compensating_set(
    request: FindCompensatingSetRequest,
    background_tasks: BackgroundTasks,
    service: ZeroSumService = Depends(get_zerosum_service),
) -> CompensatingSetResponse:
    """Find compensating set."""
    try:
        logger.info(f"Finding compensating set for: {request.target_set_id}")

        result = await service.find_compensating_set(
            target_set_id=request.target_set_id,
            method=request.method,
            tolerance=request.tolerance,
            max_iterations=request.max_iterations,
            search_space=request.search_space,
        )

        # Log in background task
        background_tasks.add_task(
            service.log_operation,
            operation_type="find_compensating",
            operation_id=result.operation_id,
            result=result.model_dump(mode="json"),
        )

        logger.info(f"Search completed: {result.operation_id}, status: {result.status}")
        return result

    except ValueError as e:
        logger.error(f"Validation error in compensating set search: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in compensating set search: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/operations/validate", response_model=ValidationResponse)
async def validate_zero_sum(
    request: ValidateZeroSumRequest,
    service: ZeroSumService = Depends(get_zerosum_service),
) -> ValidationResponse:
    """Validate zero-sum."""
    try:
        logger.info(f"Validating zero-sum for sets: {request.set_ids}")

        result = await service.validate_zero_sum(
            set_ids=request.set_ids,
            tolerance=request.tolerance,
            detailed_analysis=request.include_details,
        )

        logger.info(
            f"Validation completed: {result.validation_id}, result: {result.is_valid}"
        )
        return result

    except ValueError as e:
        logger.error(f"Validation error in zero-sum check: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in zero-sum validation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/operations/batch", response_model=BatchOperationResponse)
async def perform_batch_operations(
    request: BatchOperationRequest,
    service: ZeroSumService = Depends(get_zerosum_service),
) -> BatchOperationResponse:
    """Perform batch operations."""
    try:
        logger.info(f"Performing batch of {len(request.operations)} operations")

        result = await service.perform_batch_operations(
            operations=request.operations,
            parallel=request.parallel,
            max_workers=request.max_workers,
            stop_on_error=request.stop_on_error,
            timeout=request.timeout,
        )

        logger.info(
            f"Batch completed: {result.batch_id}, successful: {result.successful_operations}/{result.total_operations}"
        )
        return result

    except ValueError as e:
        logger.error(f"Validation error in batch operations: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in batch operations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/analysis/convergence", response_model=ConvergenceAnalysisResponse)
async def analyze_convergence(
    request: ConvergenceAnalysisRequest,
    service: ZeroSumService = Depends(get_zerosum_service),
) -> ConvergenceAnalysisResponse:
    """Analyze convergence of infinite set."""
    try:
        logger.info(f"Analyzing convergence for set: {request.set_id}")

        result = await service.analyze_convergence(
            set_id=request.set_id,
            max_terms=request.max_terms,
            analysis_methods=request.analysis_methods,
        )

        logger.info(
            f"Analysis completed for {request.set_id}: {result.convergence_type}"
        )
        return result

    except ValueError as e:
        logger.error(f"Validation error in convergence analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in convergence analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats() -> CacheStatsResponse:
    """Get cache statistics."""
    try:
        cache = get_global_cache()
        stats = cache.get_stats()

        return CacheStatsResponse(
            total_operations=stats["total_requests"],
            cache_hits=stats["hits"],
            cache_misses=stats["misses"],
            hit_rate=stats["hit_rate_percent"],
            cache_size=stats["size"],
            memory_usage=stats["memory_usage_mb"],
        )
    except Exception as e:
        logger.error(f"Error getting cache statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/cache")
async def clear_cache() -> Dict[str, str]:
    """Clear cache."""
    try:
        cache = get_global_cache()
        cache.clear()
        logger.info("Cache cleared")
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/sets/{set_id}")
async def delete_infinite_set(
    set_id: str, service: ZeroSumService = Depends(get_zerosum_service)
) -> Dict[str, str]:
    """Delete infinite set."""
    try:
        success = await service.delete_infinite_set(set_id)
        if not success:
            raise HTTPException(status_code=404, detail="Set not found")

        logger.info(f"Set {set_id} deleted")
        return {"message": f"Set {set_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting set {set_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
