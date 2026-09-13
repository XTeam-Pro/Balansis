from __future__ import annotations

# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.
#
# This file is part of Balansis.
# Balansis is dual-licensed under:
#   1. GNU Affero General Public License v3.0 (AGPLv3) for open-source use.
#   2. A Commercial License for proprietary and corporate use.
#
# See LICENSING.md in the project root for license selection details.
# For commercial licensing: andrew@xteam.pro
from decimal import Decimal, localcontext
from typing import List, Literal, cast

from balansis.core.absolute import AbsoluteValue


class LedgerEntry:
    def __init__(self, account: str, amount: AbsoluteValue, memo: str = "") -> None:
        self.account = account
        self.amount = amount
        self.memo = memo
        self.decimal_amount = Decimal(str(amount.to_float()))


class Ledger:
    def __init__(self) -> None:
        self.entries: List[LedgerEntry] = []

    def post_entry(self, account: str, amount: Decimal, memo: str = "") -> None:
        if not amount.is_finite():
            raise ValueError("Ledger amounts must be finite")
        sign = 1 if amount >= 0 else -1
        value = AbsoluteValue(
            magnitude=abs(float(amount)), direction=cast(Literal[-1, 1], sign)
        )
        entry = LedgerEntry(account, value, memo)
        entry.decimal_amount = amount
        self.entries.append(entry)

    def transfer(
        self, debit_account: str, credit_account: str, amount: Decimal, memo: str = ""
    ) -> None:
        self.post_entry(debit_account, amount, memo)
        self.post_entry(credit_account, amount.copy_negate(), memo)

    def decimal_balance(self, account: str | None = None) -> Decimal:
        amounts = [
            entry.decimal_amount
            for entry in self.entries
            if account is None or entry.account == account
        ]
        if not amounts:
            return Decimal(0)
        minimum_exponent = min(int(amount.as_tuple().exponent) for amount in amounts)
        maximum_adjusted = max(amount.adjusted() for amount in amounts)
        with localcontext() as context:
            context.prec = max(
                50, maximum_adjusted - minimum_exponent + len(str(len(amounts))) + 2
            )
            return sum(amounts, Decimal(0))

    def balance(self) -> AbsoluteValue:
        return AbsoluteValue.from_float(float(self.decimal_balance()))

    def account_balance(self, account: str) -> AbsoluteValue:
        return AbsoluteValue.from_float(float(self.decimal_balance(account)))
