from decimal import Decimal
from balansis.finance import Ledger

ledger = Ledger()
ledger.transfer("cash", "revenue", Decimal("123.4567890123456789"))
assert ledger.decimal_balance() == Decimal(0)
assert ledger.decimal_balance("cash") == Decimal("123.4567890123456789")
