# balansis.finance.ledger

[Source](../../balansis/finance/ledger.py) · [Reference index](index.md)

```python
class LedgerEntry:
    def __init__(self, account: str, amount: AbsoluteValue, memo: str='') -> None:
        ...
```

```python
class Ledger:
    def __init__(self) -> None:
        ...

    def post_entry(self, account: str, amount: Decimal, memo: str='') -> None:
        ...

    def transfer(self, debit_account: str, credit_account: str, amount: Decimal, memo: str='') -> None:
        ...

    def decimal_balance(self, account: str | None=None) -> Decimal:
        ...

    def balance(self) -> AbsoluteValue:
        ...

    def account_balance(self, account: str) -> AbsoluteValue:
        ...
```
