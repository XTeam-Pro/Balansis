# Copyright (c) 2024-2026 Tikhonov Andrey. All rights reserved.
# SPDX-License-Identifier: MIT (non-commercial) | Commercial use: see COMMERCIAL_LICENSE.md
from balansis.core.absolute import AbsoluteValue

def harmonic_generator(sign: int = 1):
    n = 1
    while True:
        yield AbsoluteValue(magnitude=1.0 / float(n), direction=int(sign))
        n += 1

def grandis_generator():
    d = 1
    while True:
        yield AbsoluteValue(magnitude=1.0, direction=d)
        d = -d
