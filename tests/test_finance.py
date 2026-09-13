from decimal import Decimal

from balansis.finance.ledger import Ledger


def test_double_entry_zero_sum():
    l = Ledger()
    l.transfer("Assets:Cash", "Income:Sales", Decimal("100.00"), "Sale")
    l.transfer("Expenses:Rent", "Assets:Cash", Decimal("50.00"), "Rent")
    assert l.balance().is_absolute()
    assert l.account_balance("Assets:Cash").to_float() == 50.0
    assert l.account_balance("Income:Sales").to_float() == -100.0
    assert l.account_balance("Expenses:Rent").to_float() == 50.0
