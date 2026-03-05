# Copyright (c) 2024-2026 Tikhonov Andrey. All rights reserved.
# SPDX-License-Identifier: MIT (non-commercial) | Commercial use: see COMMERCIAL_LICENSE.md
try:
    import pyarrow as pa
except ImportError:
    pa = None

from typing import List
from balansis.core.absolute import AbsoluteValue

def to_record_batch(values: List[AbsoluteValue]):
    if pa is None:
        raise ImportError("pyarrow not installed")
    mag = [v.magnitude for v in values]
    dir = [v.direction for v in values]
    arrays = [pa.array(mag, type=pa.float64()), pa.array(dir, type=pa.int8())]
    schema = pa.schema([("magnitude", pa.float64()), ("direction", pa.int8())])
    return pa.RecordBatch.from_arrays(arrays, schema.names)

def to_table(values: List[AbsoluteValue]):
    if pa is None:
        raise ImportError("pyarrow not installed")
    batch = to_record_batch(values)
    return pa.Table.from_batches([batch])

def from_table(table) -> List[AbsoluteValue]:
    if pa is None:
        raise ImportError("pyarrow not installed")
    mag = table.column("magnitude").to_pylist()
    dir = table.column("direction").to_pylist()
    return [AbsoluteValue(magnitude=float(m), direction=int(d)) for m, d in zip(mag, dir)]
