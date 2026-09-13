"""Application operations using stored series parameters and numerical results."""

import asyncio
import time
import uuid
from datetime import datetime, timezone

from ...core import ZeroSumInfiniteSet, get_global_cache, get_global_parallel_processor
from ...database.repository import InfiniteSetRepository
from ..models.requests import OperationMethod, SeriesType
from ..models.responses import (
    BatchOperationResponse,
    CompensatingSetResponse,
    ConvergenceAnalysisResponse,
    ConvergenceType,
    InfiniteSetResponse,
    OperationStatus,
    ValidationResponse,
    ZeroSumOperationResponse,
)


class ZeroSumService:
    def __init__(self):
        self.repository = InfiniteSetRepository()
        self.cache = get_global_cache()
        self.parallel_processor = get_global_parallel_processor()

    @staticmethod
    def _series(series_type, parameters):
        kind = SeriesType(series_type)
        count = parameters.get("n_terms", 1000)
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or not 0 <= count <= 100000
        ):
            raise ValueError("n_terms must be an integer between 0 and 100000")
        if kind == SeriesType.HARMONIC:
            return ZeroSumInfiniteSet.create_harmonic_series(
                count, parameters.get("p", 1.0)
            )
        if kind == SeriesType.ALTERNATING:
            return ZeroSumInfiniteSet.create_alternating_series(
                count, parameters.get("p", 1.0)
            )
        if kind == SeriesType.GEOMETRIC:
            return ZeroSumInfiniteSet.create_geometric_series(
                parameters["ratio"], count
            )
        elements = parameters.get("elements")
        if not isinstance(elements, list) or len(elements) > 100000:
            raise ValueError("Custom series requires a list of at most 100000 elements")
        return ZeroSumInfiniteSet(elements)

    async def _load(self, set_id):
        data = await self.repository.get_infinite_set(set_id)
        if data is None:
            raise ValueError(f"Set {set_id} not found")
        series = self._series(data["series_type"], data["parameters"])
        series.id = str(data["id"])
        return series

    @staticmethod
    def _response(data):
        series = ZeroSumService._series(data["series_type"], data["parameters"])
        return InfiniteSetResponse(
            id=str(data["id"]),
            name=data["name"],
            series_type=data["series_type"],
            parameters=data["parameters"],
            description=data.get("description"),
            created_at=data["created_at"],
            convergence_info=data.get("convergence_info"),
            partial_sums=[
                float(series.get_partial_sum(n))
                for n in range(1, min(10, len(series.elements)) + 1)
            ],
        )

    async def create_infinite_set(
        self, name, series_type, parameters, description=None
    ):
        series = self._series(series_type, parameters)
        set_id = str(uuid.uuid4())
        kind = SeriesType(series_type).value
        convergence = series.convergence_analysis(
            max_terms=min(1000, len(series.elements))
        )
        # Persist finite JSON data, including empty/short series analyses.
        convergence["variance"] = (
            float(convergence["variance"]) if len(series.elements) > 10 else None
        )
        convergence["is_convergent"] = bool(convergence["is_convergent"])
        await self.repository.create_infinite_set(
            set_id, name, kind, parameters, description, convergence
        )
        return self._response(
            {
                "id": set_id,
                "name": name,
                "series_type": kind,
                "parameters": parameters,
                "description": description,
                "created_at": datetime.now(timezone.utc),
                "convergence_info": convergence,
            }
        )

    async def get_infinite_set(self, set_id):
        data = await self.repository.get_infinite_set(set_id)
        return self._response(data) if data is not None else None

    async def list_infinite_sets(self, limit=100, offset=0):
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("Invalid pagination")
        return [
            self._response(data)
            for data in await self.repository.list_infinite_sets(limit, offset)
        ]

    async def perform_zero_sum_operation(
        self, set_ids, method, tolerance, max_iterations, use_cache=True
    ):
        started = time.perf_counter()
        if len(set_ids) < 2 or tolerance <= 0 or not float(tolerance) < float("inf"):
            raise ValueError(
                "At least two sets and a finite positive tolerance are required"
            )
        method = OperationMethod(method)
        series = [await self._load(identifier) for identifier in set_ids]
        cache_args = ([item.to_dict()["elements"] for item in series], method.value)
        value = self.cache.get("series_sum", *cache_args) if use_cache else None
        cached = value is not None
        if value is None:
            merged = ZeroSumInfiniteSet(
                [value for item in series for value in item.elements]
            )
            value = merged.zero_sum_operation(method=method.value)["sum"]
            if use_cache:
                self.cache.set("series_sum", value, *cache_args)
        error = abs(float(value))
        return ZeroSumOperationResponse(
            operation_id=str(uuid.uuid4()),
            status=(
                OperationStatus.SUCCESS
                if error <= tolerance
                else OperationStatus.FAILED
            ),
            result=float(value),
            method_used=method.value,
            iterations=1,
            tolerance_achieved=error,
            execution_time=time.perf_counter() - started,
            set_ids=set_ids,
            cached=cached,
            compensation_details={"residual": error},
        )

    async def find_compensating_set(
        self, target_set_id, method, tolerance, max_iterations, search_space=None
    ):
        started = time.perf_counter()
        target = await self._load(target_set_id)
        method = OperationMethod(method)
        result = target.find_compensating_set(method.value)
        residual = abs(float(target.zero_sum_operation(result)))
        stored = await self.create_infinite_set(
            result.name,
            SeriesType.CUSTOM,
            {"elements": [str(x) for x in result.elements]},
        )
        return CompensatingSetResponse(
            operation_id=str(uuid.uuid4()),
            status=(
                OperationStatus.SUCCESS
                if residual <= tolerance
                else OperationStatus.FAILED
            ),
            target_set_id=target_set_id,
            compensating_set=stored,
            compensation_quality=1.0 / (1.0 + residual),
            method_used=method.value,
            search_iterations=1,
            execution_time=time.perf_counter() - started,
            search_details={"residual": residual},
        )

    async def validate_zero_sum(self, set_ids, tolerance, detailed_analysis=False):
        result = await self.perform_zero_sum_operation(
            set_ids, OperationMethod.DIRECT, tolerance, 1, False
        )
        return ValidationResponse(
            validation_id=result.operation_id,
            is_valid=abs(result.result) <= tolerance,
            sum_value=result.result,
            tolerance_used=tolerance,
            set_ids=set_ids,
            execution_time=result.execution_time,
            partial_sums_analysis=(
                [
                    {
                        "set_id": identifier,
                        "partial_sums": [
                            float((await self._load(identifier)).get_partial_sum(n))
                            for n in range(1, 11)
                        ],
                    }
                    for identifier in set_ids
                ]
                if detailed_analysis
                else None
            ),
        )

    async def perform_batch_operations(
        self,
        operations,
        parallel=True,
        max_workers=None,
        timeout=None,
        stop_on_error=False,
    ):
        started = time.perf_counter()
        workers = max_workers if max_workers is not None else 4
        if workers <= 0 or (timeout is not None and timeout <= 0):
            raise ValueError("Worker count and timeout must be positive")
        semaphore = asyncio.Semaphore(workers)
        methods = {
            "create": self.create_infinite_set,
            "zero_sum": self.perform_zero_sum_operation,
            "find_compensating": self.find_compensating_set,
            "validate": self.validate_zero_sum,
        }

        async def run(operation):
            async with semaphore:
                try:
                    result = await methods[operation["type"]](**operation["params"])
                    return {
                        "status": "success",
                        "result": result.model_dump(mode="json"),
                    }
                except Exception as exc:
                    return {"status": "failed", "error": str(exc)}

        async def execute():
            if parallel and not stop_on_error:
                return await asyncio.gather(
                    *(run(operation) for operation in operations)
                )
            results = []
            for operation in operations:
                result = await run(operation)
                results.append(result)
                if stop_on_error and result["status"] == "failed":
                    break
            return results

        results = await asyncio.wait_for(execute(), timeout=timeout)
        successes = sum(result["status"] == "success" for result in results)
        return BatchOperationResponse(
            batch_id=str(uuid.uuid4()),
            total_operations=len(results),
            successful_operations=successes,
            failed_operations=len(results) - successes,
            results=results,
            execution_time=time.perf_counter() - started,
            parallel_execution=parallel and not stop_on_error,
        )

    async def analyze_convergence(self, set_id, max_terms, analysis_methods):
        series = await self._load(set_id)
        analysis = series.convergence_analysis(max_terms)
        sums = analysis["partial_sums"]
        return ConvergenceAnalysisResponse(
            set_id=set_id,
            convergence_type=ConvergenceType.UNKNOWN,
            analysis_methods={
                "partial_sums": {"terms": len(sums), "final_sum": analysis["final_sum"]}
            },
            partial_sums=sums,
            convergence_plot_data={"x": list(range(1, len(sums) + 1)), "y": sums},
        )

    async def delete_infinite_set(self, set_id):
        self.cache.delete(f"set:{set_id}")
        return await self.repository.delete_infinite_set(set_id)

    async def log_operation(self, operation_type, operation_id, result):
        await self.repository.log_operation(
            operation_id,
            operation_type,
            result,
            result.get("result"),
            result.get("status", "unknown"),
            result.get("execution_time", 0.0),
        )
