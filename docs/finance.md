# Decimal ledger

`Ledger.post_entry(account, amount, memo="")` accepts a `Decimal` amount.
`transfer(debit_account, credit_account, amount)` records opposing entries.
`decimal_balance(account=None)` sums the submitted decimals with a local
precision chosen from their exponents and digit counts.

`balance()` and `account_balance(account)` return the result as `AbsoluteValue`.
Each entry exposes `account`, `amount`, `decimal_amount` and `memo`.

```python
from decimal import Decimal
from balansis.finance import Ledger

ledger = Ledger()
ledger.transfer("cash", "revenue", Decimal("123.4567890123456789"))
assert ledger.decimal_balance() == Decimal(0)
assert ledger.decimal_balance("cash") == Decimal("123.4567890123456789")
```

[Source](../balansis/finance/ledger.py).
