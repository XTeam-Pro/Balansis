# Installation and first calculations

Run from the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install .
python -m pip install ./native
balansis --version
balansis doctor
balansis add 2 -1 --json
```

Install optional integrations with `python -m pip install '.[pandas,arrow,plot,torch]'`.
A C compiler is required when building `native/`. Building the Python package
alone produces a platform-independent wheel.

```python
from balansis import AbsoluteValue, EternalRatio, Operations

left = AbsoluteValue(magnitude=6.0, direction=1)
right = AbsoluteValue.from_float(2.0)
ratio, factor = Operations.compensated_divide(left, right)
assert ratio.numerical_value() == 3.0
assert EternalRatio.from_values(-6.0, 2.0).numerical_value() == -3.0
```

Use named fields with Pydantic models. `to_float()` converts an `AbsoluteValue`;
`numerical_value()` evaluates an `EternalRatio`. [API reference](reference/index.md).
