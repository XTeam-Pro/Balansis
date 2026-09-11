# TNSIM

TNSIM stores finite prefixes of harmonic, alternating, geometric and custom
series. Elements are `Decimal` values. `ZeroSumInfiniteSet` computes partial
sums, pairs a series with its negation, serializes decimal strings and reports
sampled partial-sum statistics.

```python
from tnsim.core import ZeroSumInfiniteSet

series = ZeroSumInfiniteSet.create_geometric_series(ratio=0.5, n_terms=10)
opposite = series.find_compensating_set("direct")
assert series.zero_sum_operation(opposite) == 0
assert len(series.elements) == 10
```

`TNSIMCache` provides bounded LRU caching, expiry and JSON persistence.
`ParallelTNSIM` evaluates pairs using a thread pool and offers process-based
compensation and convergence operations. Use its context manager to close pools.
`ZeroSumAttention` and `ZeroSumTransformerBlock` integrate with PyTorch autograd.

## Run the API

From the repository root:

```bash
python -m pip install './tnsim[api]'
export TNSIM_DB_PASSWORD="$(python -c 'import secrets; print(secrets.token_hex(24))')"
docker compose -f tnsim/docker-compose.yml up --build
```

The Compose configuration creates PostgreSQL and serves the API at
`http://127.0.0.1:8000`. Interactive API documentation is served at `/docs`.
For a separately configured database, set `DATABASE_URL` and initialize it using
`tnsim/database/init.sql`, then run `tnsim-server`.

The `/api/zerosum` routes create, retrieve, list and delete sets; perform
zero-sum, compensation, validation and batch operations; inspect convergence
samples; and inspect or clear the cache. `/health` checks PostgreSQL access.

Custom series use `parameters.elements`; generated series use `parameters.n_terms`
and `p` or `ratio`. API results report computed residuals. Batch operations
accept `parallel`, `max_workers`, `timeout` and `stop_on_error`.
[Application code](../tnsim/api).
