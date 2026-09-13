import asyncio
import json
from decimal import Decimal, getcontext

import pytest

from tnsim.core import ParallelTNSIM, TNSIMCache, ZeroSumInfiniteSet


@pytest.mark.parametrize("method", ["direct", "compensated", "stabilized"])
def test_sum_and_compensating_set(method):
    source = ZeroSumInfiniteSet([1, 2, 3])
    opposite = source.find_compensating_set()
    assert source.zero_sum_operation(opposite, method) == 0
    assert source.validate_zero_sum()["is_zero_sum"]
    assert source.get_partial_sum(2, start=1) == 5


@pytest.mark.parametrize("value", [float("inf"), float("nan"), Decimal("NaN")])
def test_nonfinite_rejected(value):
    with pytest.raises(ValueError):
        ZeroSumInfiniteSet([value])


@pytest.mark.parametrize("method", ["direct", "iterative", "adaptive"])
def test_compensation_after_mutation(method):
    source = ZeroSumInfiniteSet([1, 2])
    source.find_compensating_set(method)
    source.elements[0] = Decimal(3)
    opposite = source.find_compensating_set(method)
    assert source.zero_sum_operation(opposite) == 0


def test_lossless_serialization():
    source = ZeroSumInfiniteSet([Decimal("1.12345678901234567890123456789")])
    result = ZeroSumInfiniteSet.from_dict(json.loads(json.dumps(source.to_dict())))
    assert list(result.elements) == list(source.elements)


@pytest.mark.parametrize(
    "factory",
    [
        ZeroSumInfiniteSet.create_harmonic_series,
        ZeroSumInfiniteSet.create_alternating_series,
    ],
)
def test_series_local_decimal_precision(factory):
    original = getcontext().prec
    result = factory(5)
    assert len(result.elements) == 5
    assert getcontext().prec == original
    with pytest.raises(ValueError):
        factory(-1)


def test_cache_eviction_persistence_and_delete(tmp_path):
    cache = TNSIMCache(max_size=2, cache_dir=str(tmp_path))
    cache.set("a", Decimal("0.1234567890123456789"))
    cache.set("b", {"value": 2})
    assert cache.get("a") == Decimal("0.1234567890123456789")
    cache.set("c", 3)
    assert cache.get("b") is None
    restored = TNSIMCache(max_size=2, cache_dir=str(tmp_path))
    assert restored.get("a") == cache.get("a")
    assert restored.delete("a")
    assert restored.get("a") is None
    restored.clear()
    assert not list(tmp_path.glob("*.json"))


def test_cache_ignores_pickle_and_corrupt_json(tmp_path):
    (tmp_path / "ignored.pkl").write_bytes(b"not loaded")
    (tmp_path / "broken.json").write_text("{")
    assert TNSIMCache(cache_dir=str(tmp_path)).get_stats()["current_size"] == 0


@pytest.mark.parametrize(
    "options", [{"max_size": 0}, {"ttl_hours": 0}, {"ttl_hours": float("nan")}]
)
def test_cache_options(options):
    with pytest.raises(ValueError):
        TNSIMCache(persistent=False, **options)


def test_parallel_pairs_and_resource_cleanup():
    pairs = [(ZeroSumInfiniteSet([i]), ZeroSumInfiniteSet([-i])) for i in range(4)]
    with ParallelTNSIM(max_workers=2, chunk_size=2, use_cache=False) as processor:
        assert (
            asyncio.run(processor.parallel_zero_sum_operations(pairs))
            == [Decimal(0)] * 4
        )
        assert (
            len(
                processor.batch_operations(
                    [{"type": "validate_zero_sum", "set": pairs[0][0]}]
                )
            )
            == 1
        )
    with pytest.raises(RuntimeError):
        processor.thread_pool.submit(lambda: None)


def test_database_configuration_preserves_explicit_environment(monkeypatch):
    from urllib.parse import unquote, urlsplit

    from tnsim.database.config import get_config

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("POSTGRES_HOST", "database.example")
    monkeypatch.setenv("POSTGRES_USER", "test user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "synthetic +/@ value")
    monkeypatch.setenv("POSTGRES_DB", "test database")
    config = get_config("development")
    parsed = urlsplit(config.asyncpg_dsn)
    assert parsed.hostname == "database.example"
    assert unquote(parsed.username) == "test user"
    assert unquote(parsed.password) == "synthetic +/@ value"
    assert unquote(parsed.path) == "/test database"
    assert "synthetic" not in repr(config)
    monkeypatch.setenv("DATABASE_URL", "postgresql:///explicit")
    assert config.asyncpg_dsn == "postgresql:///explicit"
