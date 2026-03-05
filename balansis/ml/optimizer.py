# Copyright (c) 2024-2026 Tikhonov Andrey. All rights reserved.
# SPDX-License-Identifier: MIT (non-commercial) | Commercial use: see COMMERCIAL_LICENSE.md
try:
    import torch
except ImportError:
    torch = None

from balansis.core.absolute import AbsoluteValue
from balansis.core.eternity import EternalRatio

class EternalOptimizer:
    def __init__(self, params, lr: float = 1e-3):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            if torch is not None and hasattr(p, "grad") and p.grad is not None:
                g = p.grad
                num = AbsoluteValue.from_float(float(self.lr))
                den = AbsoluteValue.from_float(float(torch.linalg.norm(g))) if torch is not None else AbsoluteValue.unit_positive()
                r = EternalRatio(numerator=num, denominator=den)
                p.data = p.data - r.numerical_value() * g

if torch is not None:
    class EternalTorchOptimizer(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3):
            defaults = dict(lr=lr)
            super().__init__(params, defaults)

        def step(self, closure=None):
            loss = None
            if closure is not None:
                loss = closure()
            for group in self.param_groups:
                lr = group["lr"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    num = AbsoluteValue.from_float(float(lr))
                    den = AbsoluteValue.from_float(float(torch.linalg.norm(g)))
                    r = EternalRatio(numerator=num, denominator=den)
                    p.data = p.data - r.numerical_value() * g
            return loss
