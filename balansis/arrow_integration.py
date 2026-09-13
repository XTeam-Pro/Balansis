from __future__ import annotations

from typing import Any, Literal, cast

# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.
#
# This file is part of Balansis.
# Balansis is dual-licensed under:
#   1. GNU Affero General Public License v3.0 (AGPLv3) for open-source use.
#   2. A Commercial License for proprietary and corporate use.
#
# See LICENSING.md in the project root for license selection details.
# For commercial licensing: andrew@xteam.pro
try:
    import pyarrow as pa
except ImportError:
    pa = None

from typing import List

from balansis.core.absolute import AbsoluteValue


def to_record_batch(values: List[AbsoluteValue]) -> Any:
    if pa is None:
        raise ImportError("pyarrow not installed")
    mag = [v.magnitude for v in values]
    dir = [v.direction for v in values]
    arrays = [pa.array(mag, type=pa.float64()), pa.array(dir, type=pa.int8())]
    schema = pa.schema([("magnitude", pa.float64()), ("direction", pa.int8())])
    return pa.RecordBatch.from_arrays(arrays, schema.names)


def to_table(values: List[AbsoluteValue]) -> Any:
    if pa is None:
        raise ImportError("pyarrow not installed")
    batch = to_record_batch(values)
    return pa.Table.from_batches([batch])


def from_table(table: Any) -> List[AbsoluteValue]:
    if pa is None:
        raise ImportError("pyarrow not installed")
    mag = table.column("magnitude").to_pylist()
    dir = table.column("direction").to_pylist()
    return [
        AbsoluteValue(magnitude=float(m), direction=cast(Literal[-1, 1], int(d)))
        for m, d in zip(mag, dir)
    ]
