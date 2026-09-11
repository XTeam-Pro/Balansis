import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_postgres_api_workflow(database_url, monkeypatch, tmp_path):
    asyncpg = pytest.importorskip("asyncpg")
    monkeypatch.setenv("DATABASE_URL", database_url)
    import tnsim.database.repository as repository

    repository._db_connection = repository.DatabaseConnection()
    import tnsim.core.cache.tnsim_cache as caching

    caching._global_cache = caching.TNSIMCache(persistent=False)
    import tnsim.core.operations.parallel_tnsim as parallel

    parallel._global_parallel_processor = None

    async def initialize():
        connection = await asyncpg.connect(database_url)
        await connection.execute(Path("tnsim/database/init.sql").read_text())
        await connection.close()

    asyncio.run(initialize())
    from tnsim.api.main import app

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        identifiers = []
        for elements in ([1, 2, 3], [-1, -2, -3]):
            response = client.post(
                "/api/zerosum/sets",
                json={
                    "name": "synthetic series",
                    "series_type": "custom",
                    "parameters": {"elements": elements},
                },
            )
            assert response.status_code == 200, response.text
            identifiers.append(response.json()["id"])
        body = {"set_ids": identifiers, "method": "compensated"}
        for _ in range(2):
            response = client.post("/api/zerosum/operations/zero-sum", json=body)
            assert response.status_code == 200, response.text
            assert response.json()["result"] == 0
        assert response.json()["cached"]
        response = client.post(
            "/api/zerosum/operations/validate",
            json={"set_ids": identifiers, "include_details": True},
        )
        assert response.status_code == 200, response.text
        assert response.json()["is_valid"]
        response = client.post(
            "/api/zerosum/operations/find-compensating",
            json={"target_set_id": identifiers[0], "method": "direct"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "success"
        response = client.get("/api/zerosum/cache/stats")
        assert response.status_code == 200, response.text
        response = client.post(
            "/api/zerosum/operations/batch",
            json={
                "operations": [
                    {
                        "type": "validate",
                        "params": {"set_ids": identifiers, "tolerance": 1e-10},
                    }
                ]
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["successful_operations"] == 1
        for identifier in identifiers:
            assert client.delete("/api/zerosum/sets/" + identifier).status_code == 200
            assert client.get("/api/zerosum/sets/" + identifier).status_code == 404
