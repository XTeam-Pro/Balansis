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
from typing import Iterable, Iterator

from balansis.core.absolute import AbsoluteValue


class EternalSet:
    __slots__ = ("_source", "is_infinite", "rule_name")

    def __init__(
        self,
        source: Iterable[AbsoluteValue],
        is_infinite: bool = False,
        rule_name: str = "custom",
    ) -> None:
        self._source = source
        self.is_infinite = bool(is_infinite)
        self.rule_name = str(rule_name)

    def __iter__(self) -> Iterator[AbsoluteValue]:
        for x in self._source:
            if not isinstance(x, AbsoluteValue):
                raise TypeError("EternalSet elements must be AbsoluteValue")
            yield x

    def __repr__(self) -> str:
        return f"<EternalSet(rule={self.rule_name}, infinite={self.is_infinite})>"
