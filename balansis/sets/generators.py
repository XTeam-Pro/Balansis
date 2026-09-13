from __future__ import annotations

from typing import Iterator, Literal, cast

# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.
#
# This file is part of Balansis.
# Balansis is dual-licensed under:
#   1. GNU Affero General Public License v3.0 (AGPLv3) for open-source use.
#   2. A Commercial License for proprietary and corporate use.
#
# See LICENSING.md in the project root for license selection details.
# For commercial licensing: andrew@xteam.pro
from balansis.core.absolute import AbsoluteValue


def harmonic_generator(sign: int = 1) -> Iterator[AbsoluteValue]:
    n = 1
    while True:
        yield AbsoluteValue(
            magnitude=1.0 / float(n), direction=cast(Literal[-1, 1], int(sign))
        )
        n += 1


def grandis_generator() -> Iterator[AbsoluteValue]:
    d = 1
    while True:
        yield AbsoluteValue(magnitude=1.0, direction=cast(Literal[-1, 1], d))
        d = -d
