from decimal import Decimal

import pytest

from balansis import AbsoluteValue, Operations
from balansis.finance.ledger import Ledger


def naive_float_sum(values):
    total = 0.0
    for value in values:
        total += value
    return total


def test_readme_large_scale_aggregation_preserves_residual():
    builtin_sum_result = sum([1e16, 1.0, -1e16])
    naive_sum_result = naive_float_sum([1e16, 1.0, -1e16])
    assert builtin_sum_result in (0.0, 1.0)
    assert naive_sum_result == 0.0

    values = [
        AbsoluteValue.from_float(1e16),
        AbsoluteValue.from_float(1.0),
        AbsoluteValue.from_float(-1e16),
    ]
    result, compensation = Operations.sequence_sum(values)

    assert result == AbsoluteValue.from_float(1.0)
    assert compensation > 0.0


def test_readme_cancellation_respects_the_represented_operands():
    float_result = (1e16 + 1.0) - 1e16
    assert float_result == 0.0

    left = AbsoluteValue.from_float(1e16)
    right = AbsoluteValue.from_float(-1e16)
    result, compensation = Operations.compensated_add(left, right)

    assert result.is_absolute()
    assert result.to_float() == 0.0
    assert compensation == Operations.STABILITY_FACTOR


def test_finance_workflow_balances_to_absolute():
    ledger = Ledger()
    ledger.post_entry("cash", Decimal("250.00"))
    ledger.post_entry("cash", Decimal("-250.00"))

    balance = ledger.balance()

    assert balance.is_absolute()
    assert balance == 0.0


def test_division_contract_returns_ratio_for_valid_denominator():
    numerator = AbsoluteValue.from_float(6.0)
    denominator = AbsoluteValue.from_float(2.0)

    ratio, compensation = Operations.compensated_divide(numerator, denominator)

    assert ratio.numerical_value() == pytest.approx(3.0)
    assert ratio.signed_value() == 1.0
    assert compensation == 1.0


def test_division_contract_rejects_absolute_denominator():
    numerator = AbsoluteValue.from_float(6.0)

    with pytest.raises(ValueError, match="Cannot divide by Absolute"):
        Operations.compensated_divide(numerator, AbsoluteValue.absolute())
