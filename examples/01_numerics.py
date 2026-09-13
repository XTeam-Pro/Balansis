from balansis import B, Operations, SingularPolicy

result, factor, event = Operations.compensated_divide_policy(
    B(2), B(0), SingularPolicy.SATURATE, saturation_limit=100.0
)
assert result.numerical_value() == 100.0
assert event.saturated
