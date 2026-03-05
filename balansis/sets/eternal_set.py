# Copyright (c) 2024-2026 Tikhonov Andrey. All rights reserved.
# SPDX-License-Identifier: MIT (non-commercial) | Commercial use: see COMMERCIAL_LICENSE.md
from typing import Iterable, Iterator, Optional
from balansis.core.absolute import AbsoluteValue

class EternalSet:
    __slots__ = ("_source", "is_infinite", "rule_name")
    def __init__(self, source: Iterable[AbsoluteValue], is_infinite: bool = False, rule_name: str = "custom"):
        self._source = source
        self.is_infinite = bool(is_infinite)
        self.rule_name = str(rule_name)

    def __iter__(self) -> Iterator[AbsoluteValue]:
        for x in self._source:
            if not isinstance(x, AbsoluteValue):
                raise TypeError("EternalSet elements must be AbsoluteValue")
            yield x

    def __repr__(self):
        return f"<EternalSet(rule={self.rule_name}, infinite={self.is_infinite})>"
