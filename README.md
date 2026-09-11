# Balansis

Balansis 1.2.0 provides signed-magnitude values, explicit singular arithmetic,
compensated summation, exact accumulation of binary64 dot products, matrix
factorizations and finite algebraic operations.

## Install from source

```bash
python -m pip install .
python -m pip install ./native
balansis doctor
```

The Python package uses NumPy and Pydantic on Python 3.10 or newer. The optional
`balansis-kernels` package builds C11 kernels for CPython. Array functions select
the installed kernel automatically; `backend="python"` selects the Python path.

## Compute

```python
from balansis import B, sum_array, dot_array, gram_pair

value, correction = sum_array([1e16, 1.0, -1e16])
assert value.to_float() == 1.0
assert dot_array([1e16, 1.0, -1e16], [1.0, 1.0, 1.0]) == 1.0
assert gram_pair([1.0, 2.0], [3.0, 4.0]) == (5.0, 25.0, 11.0)
assert (B(3) + B(-2)).to_float() == 1.0
```

[Documentation](docs/index.md) · [API reference](docs/reference/index.md) ·
[Examples](examples/README.md) · [Release](docs/release.md) ·
[Contributing](CONTRIBUTING.md)

## License

Code is available under AGPL-3.0-only or a separate commercial license.
See [LICENSE](LICENSE), [LICENSING.md](LICENSING.md) and
[COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).
