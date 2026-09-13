# TNSIM

TNSIM provides finite series computations, Decimal serialization, bounded JSON
caching, parallel operations and a PostgreSQL-backed FastAPI application.

Install from this directory with `python -m pip install '.[api]'`.
Set `DATABASE_URL`, initialize PostgreSQL using `database/init.sql`, and run
`tnsim-server`. The default address is `127.0.0.1:8000`; OpenAPI UI is `/docs`.

`ZeroSumInfiniteSet` accepts Decimal-compatible elements and generates harmonic,
alternating and geometric prefixes. `TNSIMCache` provides LRU eviction and TTL.
Use `ParallelTNSIM` as a context manager. Install `.[torch]` for attention modules.
