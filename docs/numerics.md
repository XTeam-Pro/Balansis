# Values and arithmetic

`AbsoluteValue` stores a finite nonnegative magnitude and a direction of `-1` or
`1`. Its numeric value is their product. Instances are immutable. Addition and
subtraction compare magnitudes for opposing signs; exact cancellation returns
positive zero. Scalar multiplication and division return `AbsoluteValue`.
`exp`, `log`, `sin`, `cos` and `tan` evaluate the signed numeric argument.

`EternalRatio` stores an `AbsoluteValue` numerator and a nonzero denominator.
`value()` returns the magnitude ratio, `signed_value()` returns its sign, and
`numerical_value()` returns the signed quotient. Integer powers preserve the
sign according to parity, including negative exponents. Fractional powers of a
negative value raise `ValueError`.

## Compensation operations

`Operations.sequence_sum` applies Neumaier accumulation and returns a value
plus the absolute correction divided by `COMPENSATION_THRESHOLD`. `sum_array`
provides the array entry point for this calculation. `use_compensation=False`
selects sequential ordinary addition.

`Operations.compensated_add` retains a near-cancellation residual for large
operands and maps other nonzero results below `1e-15` to zero.
`compensated_multiply` maps products below `1e-15` to zero and saturates products
above `1e100`. Its second return value records the applied compensation.
`compensated_divide` constructs a ratio and rejects a zero denominator.

`Compensator` applies the configured stability, balance and overflow policies
and records operations in a bounded history. `CompensationStrategy` selects
thresholds; `get_compensation_summary()` and `get_singular_telemetry()` expose
records. The record timestamp is the operation counter.

## Singular states

`ExtendedRatio` has `finite`, `infinite` and `indeterminate` states. Direct
construction through `from_division` maps nonzero/zero to signed infinity and
zero/zero to indeterminate. Opposing infinities add to indeterminate; zero times
infinity is indeterminate. Inversion of infinity returns zero. The ratio-division
operator maps division by a finite zero ratio to indeterminate.

```python
from balansis import B, Operations, SingularPolicy

result, factor, event = Operations.compensated_divide_policy(
    B(2), B(0), SingularPolicy.SATURATE, saturation_limit=100.0
)
assert result.numerical_value() == 100.0
assert event.saturated
```

`raise` rejects a singular result, `propagate` preserves its state, and `saturate`
replaces infinity with a finite signed limit. An indeterminate result remains
indeterminate under saturation. [Definitions](../balansis/core/eternity.py).
