"""Repository for working with infinite sets database."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import asyncpg
except ImportError:  # pragma: no cover - optional dependency
    asyncpg = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Database connection manager."""

    def __init__(self):
        self.pool = None
        self._initialize_lock = asyncio.Lock()
        self.connection_string = self._get_connection_string()

    def _get_connection_string(self) -> str:
        """Get connection string from environment variables."""
        from .config import DatabaseConfig

        return DatabaseConfig().asyncpg_dsn

    async def initialize(self):
        """Initialize connection pool."""
        if asyncpg is None:
            raise RuntimeError(
                "asyncpg is required for TNSIM database operations. "
                "Install the database extras to enable repository access."
            )
        async with self._initialize_lock:
            if self.pool is not None:
                return
            try:
                self.pool = await asyncpg.create_pool(
                    self.connection_string, min_size=5, max_size=20, command_timeout=60
                )
                logger.info("Database connection pool initialized")
            except Exception as e:
                logger.error(f"Database initialization error: {e}")
                raise

    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Database connection pool closed")

    async def get_connection(self):
        """Get connection from pool."""
        if not self.pool:
            await self.initialize()
        return self.pool.acquire()


# Global connection instance
_db_connection = DatabaseConnection()


async def get_db_connection():
    """Get global database connection."""
    return await _db_connection.get_connection()


class InfiniteSetRepository:
    """Repository for operations with infinite sets."""

    def __init__(self):
        self.db = _db_connection

    async def create_infinite_set(
        self,
        set_id: str,
        name: str,
        series_type: str,
        parameters: Dict[str, Any],
        description: Optional[str] = None,
        convergence_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create new infinite set."""

        query = """
        INSERT INTO infinite_sets (
            id, name, series_type, parameters, description, 
            convergence_info, created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $7)
        RETURNING id
        """

        async with await self.db.get_connection() as conn:
            try:
                result = await conn.fetchval(
                    query,
                    set_id,
                    name,
                    series_type,
                    json.dumps(parameters),
                    description,
                    json.dumps(convergence_info) if convergence_info else None,
                    datetime.now(timezone.utc),
                )
                logger.info(f"Created infinite set: {set_id}")
                return result
            except Exception as e:
                logger.error(f"Error creating set {set_id}: {e}")
                raise

    async def get_infinite_set(self, set_id: str) -> Optional[Dict[str, Any]]:
        """Get infinite set by ID."""

        query = """
        SELECT id, name, series_type, parameters, description,
               convergence_info, created_at, updated_at
        FROM infinite_sets
        WHERE id = $1
        """

        async with await self.db.get_connection() as conn:
            try:
                row = await conn.fetchrow(query, set_id)
                if row:
                    return {
                        "id": row["id"],
                        "name": row["name"],
                        "series_type": row["series_type"],
                        "parameters": (
                            json.loads(row["parameters"]) if row["parameters"] else {}
                        ),
                        "description": row["description"],
                        "convergence_info": (
                            json.loads(row["convergence_info"])
                            if row["convergence_info"]
                            else {}
                        ),
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    }
                return None
            except Exception as e:
                logger.error(f"Error getting set {set_id}: {e}")
                raise

    async def list_infinite_sets(
        self, limit: int = 100, offset: int = 0, series_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get list of infinite sets."""

        base_query = """
        SELECT id, name, series_type, parameters, description,
               convergence_info, created_at, updated_at
        FROM infinite_sets
        """

        conditions = []
        params = []
        param_count = 0

        if series_type:
            param_count += 1
            conditions.append(f"series_type = ${param_count}")
            params.append(series_type)

        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        base_query += f" ORDER BY created_at DESC LIMIT ${param_count + 1} OFFSET ${param_count + 2}"
        params.extend([limit, offset])

        async with await self.db.get_connection() as conn:
            try:
                rows = await conn.fetch(base_query, *params)
                results = []
                for row in rows:
                    results.append(
                        {
                            "id": row["id"],
                            "name": row["name"],
                            "series_type": row["series_type"],
                            "parameters": (
                                json.loads(row["parameters"])
                                if row["parameters"]
                                else {}
                            ),
                            "description": row["description"],
                            "convergence_info": (
                                json.loads(row["convergence_info"])
                                if row["convergence_info"]
                                else {}
                            ),
                            "created_at": row["created_at"],
                            "updated_at": row["updated_at"],
                        }
                    )
                return results
            except Exception as e:
                logger.error(f"Error getting sets list: {e}")
                raise

    async def update_infinite_set(
        self,
        set_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        convergence_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update infinite set."""

        updates = []
        params = []
        param_count = 0

        if name is not None:
            param_count += 1
            updates.append(f"name = ${param_count}")
            params.append(name)

        if description is not None:
            param_count += 1
            updates.append(f"description = ${param_count}")
            params.append(description)

        if convergence_info is not None:
            param_count += 1
            updates.append(f"convergence_info = ${param_count}")
            params.append(json.dumps(convergence_info))

        if not updates:
            return False

        param_count += 1
        updates.append(f"updated_at = ${param_count}")
        params.append(datetime.now(timezone.utc))

        param_count += 1
        params.append(set_id)

        query = f"""
        UPDATE infinite_sets
        SET {', '.join(updates)}
        WHERE id = ${param_count}
        """

        async with await self.db.get_connection() as conn:
            try:
                result = await conn.execute(query, *params)
                return result == "UPDATE 1"
            except Exception as e:
                logger.error(f"Error updating set {set_id}: {e}")
                raise

    async def delete_infinite_set(self, set_id: str) -> bool:
        """Delete infinite set."""

        # First delete related elements
        await self.delete_set_elements(set_id)

        query = "DELETE FROM infinite_sets WHERE id = $1"

        async with await self.db.get_connection() as conn:
            try:
                result = await conn.execute(query, set_id)
                logger.info(f"Deleted infinite set: {set_id}")
                return result == "DELETE 1"
            except Exception as e:
                logger.error(f"Error deleting set {set_id}: {e}")
                raise

    async def save_set_elements(
        self, set_id: str, elements: List[Dict[str, Any]]
    ) -> int:
        """Save set elements."""

        if not elements:
            return 0

        query = """
        INSERT INTO set_elements (set_id, position, value, computed_at)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (set_id, position) DO UPDATE SET
            value = EXCLUDED.value,
            computed_at = EXCLUDED.computed_at
        """

        async with await self.db.get_connection() as conn:
            try:
                async with conn.transaction():
                    count = 0
                    for element in elements:
                        await conn.execute(
                            query,
                            set_id,
                            element["position"],
                            element["value"],
                            element.get("computed_at", datetime.now(timezone.utc)),
                        )
                        count += 1

                    logger.info(f"Saved {count} elements for set {set_id}")
                    return count
            except Exception as e:
                logger.error(f"Error saving elements for {set_id}: {e}")
                raise

    async def get_set_elements(
        self, set_id: str, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get set elements."""

        query = """
        SELECT position, value, computed_at
        FROM set_elements
        WHERE set_id = $1
        ORDER BY position
        LIMIT $2 OFFSET $3
        """

        async with await self.db.get_connection() as conn:
            try:
                rows = await conn.fetch(query, set_id, limit, offset)
                return [
                    {
                        "position": row["position"],
                        "value": float(row["value"]),
                        "computed_at": row["computed_at"],
                    }
                    for row in rows
                ]
            except Exception as e:
                logger.error(f"Error getting elements for {set_id}: {e}")
                raise

    async def delete_set_elements(self, set_id: str) -> int:
        """Delete set elements."""

        query = "DELETE FROM set_elements WHERE set_id = $1"

        async with await self.db.get_connection() as conn:
            try:
                result = await conn.execute(query, set_id)
                count = int(result.split()[-1]) if result.startswith("DELETE") else 0
                logger.info(f"Deleted {count} elements for set {set_id}")
                return count
            except Exception as e:
                logger.error(f"Error deleting elements for {set_id}: {e}")
                raise

    async def save_compensation_pair(
        self,
        set1_id: str,
        set2_id: str,
        compensation_quality: float,
        method_used: str,
        tolerance: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save compensation pair."""

        query = """
        INSERT INTO compensation_pairs (
            set1_id, set2_id, compensation_quality, method_used,
            tolerance, metadata, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING id
        """

        async with await self.db.get_connection() as conn:
            try:
                result = await conn.fetchval(
                    query,
                    set1_id,
                    set2_id,
                    compensation_quality,
                    method_used,
                    tolerance,
                    json.dumps(metadata) if metadata else None,
                    datetime.now(timezone.utc),
                )
                logger.info(f"Saved compensation pair: {set1_id} <-> {set2_id}")
                return result
            except Exception as e:
                logger.error(f"Error saving compensation pair: {e}")
                raise

    async def get_compensation_pairs(
        self, set_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get compensation pairs for set."""

        query = """
        SELECT id, set1_id, set2_id, compensation_quality, method_used,
               tolerance, metadata, created_at
        FROM compensation_pairs
        WHERE set1_id = $1 OR set2_id = $1
        ORDER BY compensation_quality DESC, created_at DESC
        LIMIT $2
        """

        async with await self.db.get_connection() as conn:
            try:
                rows = await conn.fetch(query, set_id, limit)
                results = []
                for row in rows:
                    results.append(
                        {
                            "id": row["id"],
                            "set1_id": row["set1_id"],
                            "set2_id": row["set2_id"],
                            "compensation_quality": float(row["compensation_quality"]),
                            "method_used": row["method_used"],
                            "tolerance": float(row["tolerance"]),
                            "metadata": (
                                json.loads(row["metadata"]) if row["metadata"] else {}
                            ),
                            "created_at": row["created_at"],
                        }
                    )
                return results
            except Exception as e:
                logger.error(f"Error getting compensation pairs for {set_id}: {e}")
                raise

    async def log_operation(
        self,
        operation_id: str,
        operation_type: str,
        parameters: Dict[str, Any],
        result: Any,
        status: str,
        execution_time: float,
        error_message: Optional[str] = None,
    ) -> None:
        """Log operation."""

        query = """
        INSERT INTO operation_logs (
            operation_id, operation_type, parameters, result,
            status, execution_time, error_message, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """

        async with await self.db.get_connection() as conn:
            try:
                await conn.execute(
                    query,
                    operation_id,
                    operation_type,
                    json.dumps(parameters),
                    json.dumps(result) if result is not None else None,
                    status,
                    execution_time,
                    error_message,
                    datetime.now(timezone.utc),
                )
            except Exception as e:
                logger.error(f"Error logging operation {operation_id}: {e}")
                # Don't raise exception to avoid disrupting main operation

    async def get_operation_logs(
        self,
        operation_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get operation logs."""

        base_query = """
        SELECT operation_id, operation_type, parameters, result,
               status, execution_time, error_message, created_at
        FROM operation_logs
        """

        conditions = []
        params = []
        param_count = 0

        if operation_type:
            param_count += 1
            conditions.append(f"operation_type = ${param_count}")
            params.append(operation_type)

        if status:
            param_count += 1
            conditions.append(f"status = ${param_count}")
            params.append(status)

        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        base_query += f" ORDER BY created_at DESC LIMIT ${param_count + 1} OFFSET ${param_count + 2}"
        params.extend([limit, offset])

        async with await self.db.get_connection() as conn:
            try:
                rows = await conn.fetch(base_query, *params)
                results = []
                for row in rows:
                    results.append(
                        {
                            "operation_id": row["operation_id"],
                            "operation_type": row["operation_type"],
                            "parameters": (
                                json.loads(row["parameters"])
                                if row["parameters"]
                                else {}
                            ),
                            "result": (
                                json.loads(row["result"]) if row["result"] else None
                            ),
                            "status": row["status"],
                            "execution_time": float(row["execution_time"]),
                            "error_message": row["error_message"],
                            "created_at": row["created_at"],
                        }
                    )
                return results
            except Exception as e:
                logger.error(f"Error getting operation logs: {e}")
                raise

    async def health_check(self) -> Dict[str, Any]:
        """Database health check."""

        try:
            async with await self.db.get_connection() as conn:
                # Simple query to check connection
                result = await conn.fetchval("SELECT 1")

                # Check main tables
                tables_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('infinite_sets', 'set_elements', 'compensation_pairs', 'operation_logs')
                """

                tables = await conn.fetch(tables_query)
                existing_tables = [row["table_name"] for row in tables]

                # Table statistics
                stats = {}
                for table in existing_tables:
                    count_query = f"SELECT COUNT(*) FROM {table}"
                    count = await conn.fetchval(count_query)
                    stats[table] = count

                return {
                    "status": "healthy",
                    "connection": "ok",
                    "tables": existing_tables,
                    "statistics": stats,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

        except Exception as e:
            logger.error(f"Database health check error: {e}")
            return {
                "status": "unhealthy",
                "connection": "failed",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


# Functions for database connection initialization and closing
async def initialize_database():
    """Initialize database connection."""
    await _db_connection.initialize()


async def close_database():
    """Close database connection."""
    await _db_connection.close()
