# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.
#
# This file is part of Balansis.
# Balansis is dual-licensed under:
#   1. GNU Affero General Public License v3.0 (AGPLv3) for open-source use.
#   2. A Commercial License for proprietary and corporate use.
#
# See LICENSING.md in the project root for license selection details.
# For commercial licensing: andrew@xteam.pro
"""ACT-aware optimizers for neural-network training.

Two optimizer flavours are provided:

* :class:`EternalOptimizer` — plain SGD with optional momentum / weight decay,
  scaling each step through an :class:`EternalRatio` so the update magnitude
  is normalised by gradient norm. PyTorch-agnostic; falls back to a no-op
  when no parameter has a ``grad`` attribute.
* :class:`AdaptiveEternalOptimizer` — Adam-like adaptive optimizer with
  ACT-stable accumulators, optional gradient clipping, linear warmup, and
  cosine decay.

When ``torch`` is available, :class:`EternalTorchOptimizer` is exported as
well; it is a thin :class:`torch.optim.Optimizer` subclass mirroring the
SGD-with-momentum semantics so it can be dropped into PyTorch training loops.
"""

from __future__ import annotations

import importlib
import math
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple

try:
    torch: Any = importlib.import_module("torch")
except ImportError:  # pragma: no cover - torch is optional
    torch = None

from balansis.core.absolute import AbsoluteValue
from balansis.core.eternity import (
    EternalRatio,
    ExtendedRatio,
    SingularArithmeticEvent,
    SingularPolicy,
)
from balansis.core.operations import Operations


def _validate_lr(lr: float) -> None:
    if not isinstance(lr, (int, float)) or lr < 0.0 or math.isnan(lr) or math.isinf(lr):
        raise ValueError(f"Invalid learning rate: {lr}")


def _validate_momentum(momentum: float) -> None:
    if not (0.0 <= momentum < 1.0):
        raise ValueError(f"Invalid momentum: {momentum}")


def _validate_weight_decay(weight_decay: float) -> None:
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError(f"Invalid weight_decay: {weight_decay}")


def _validate_max_grad_norm(value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"Invalid max_grad_norm: {value}")


def _validate_warmup(value: int) -> None:
    if value < 0:
        raise ValueError(f"Invalid warmup_steps: {value}")


def _validate_beta(name: str, value: float) -> None:
    if not (0.0 <= value < 1.0):
        raise ValueError(f"Invalid {name}: {value}")


def _extended_division_state(numerator: float, denominator: float) -> ExtendedRatio:
    ratio, _ = Operations.compensated_divide_extended(
        AbsoluteValue.from_float(numerator),
        AbsoluteValue.from_float(denominator),
    )
    return ratio


def _extended_division_policy(
    numerator: float,
    denominator: float,
    policy: SingularPolicy | str,
    saturation_limit: float,
) -> tuple[ExtendedRatio, Optional[SingularArithmeticEvent]]:
    ratio, _, event = Operations.compensated_divide_policy(
        AbsoluteValue.from_float(numerator),
        AbsoluteValue.from_float(denominator),
        policy=policy,
        saturation_limit=saturation_limit,
    )
    return ratio, event


class EternalOptimizer:
    """ACT-normalised gradient descent with optional momentum + weight decay.

    Parameters are updated as ``p ← p - (lr / ||g||) g + momentum_buffer``.
    The ``lr / ||g||`` term is computed through :class:`EternalRatio`, which
    keeps the scaling stable when the gradient norm is very small.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        singular_policy: SingularPolicy | str = SingularPolicy.PROPAGATE,
        saturation_limit: float = 1e12,
    ) -> None:
        _validate_lr(lr)
        _validate_momentum(momentum)
        _validate_weight_decay(weight_decay)

        self.params: List[Any] = list(params)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.singular_policy = SingularPolicy(singular_policy)
        self.saturation_limit = float(saturation_limit)
        self.state: Dict[int, Dict[str, Any]] = {}
        self.last_scale_state: Optional[ExtendedRatio] = None
        self.last_scale_event: Optional[SingularArithmeticEvent] = None

    def _get_state(self, key: int) -> Dict[str, Any]:
        state = self.state.get(key)
        if state is None:
            state = {"momentum_buffer": None, "step": 0}
            self.state[key] = state
        return state

    def scale_state(self, grad_norm: float) -> ExtendedRatio:
        """Return the ExtendedRatio state for ``lr / grad_norm``."""
        return _extended_division_state(self.lr, grad_norm)

    def scale_policy_state(
        self,
        grad_norm: float,
        policy: SingularPolicy | str | None = None,
    ) -> tuple[ExtendedRatio, Optional[SingularArithmeticEvent]]:
        """Return policy-resolved scale state and optional singular event."""
        return _extended_division_policy(
            self.lr,
            grad_norm,
            policy or self.singular_policy,
            self.saturation_limit,
        )

    def step(self, closure: Optional[Any] = None) -> Optional[Any]:
        loss = closure() if closure is not None else None
        if torch is None:
            return loss
        for p in self.params:
            if not hasattr(p, "grad") or p.grad is None:
                continue
            g = p.grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data

            grad_norm = float(torch.linalg.norm(g))
            self.last_scale_state, self.last_scale_event = self.scale_policy_state(
                grad_norm
            )
            if grad_norm == 0.0:
                continue
            # ACT-normalised step size: lr / ||g||
            num = AbsoluteValue.from_float(self.lr)
            den = AbsoluteValue.from_float(grad_norm)
            scale = EternalRatio(numerator=num, denominator=den).numerical_value()

            update = scale * g

            state = self._get_state(id(p))
            if self.momentum > 0.0:
                buf = state["momentum_buffer"]
                if buf is None:
                    buf = update.clone()
                else:
                    buf = self.momentum * buf + update
                state["momentum_buffer"] = buf
                update = buf
            state["step"] += 1
            p.data = p.data - update
        return loss


if torch is not None:

    class EternalTorchOptimizer(torch.optim.Optimizer):
        """PyTorch :class:`Optimizer` subclass wrapping :class:`EternalOptimizer`.

        Supports parameter groups, momentum buffers, and weight decay. Each
        step normalises the update by gradient norm via an :class:`EternalRatio`.
        """

        def __init__(
            self,
            params: Iterable[Any],
            lr: float = 1e-3,
            momentum: float = 0.0,
            weight_decay: float = 0.0,
            singular_policy: SingularPolicy | str = SingularPolicy.PROPAGATE,
            saturation_limit: float = 1e12,
        ) -> None:
            _validate_lr(lr)
            _validate_momentum(momentum)
            _validate_weight_decay(weight_decay)
            self.singular_policy = SingularPolicy(singular_policy)
            self.saturation_limit = float(saturation_limit)
            defaults = {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}
            super().__init__(params, defaults)
            self.last_scale_state: Optional[ExtendedRatio] = None
            self.last_scale_event: Optional[SingularArithmeticEvent] = None

        def step(self, closure: Optional[Callable[[], Any]] = None) -> Any:
            loss = closure() if closure is not None else None
            for group in self.param_groups:
                lr = group["lr"]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if weight_decay != 0.0:
                        g = g + weight_decay * p.data

                    grad_norm = float(torch.linalg.norm(g))
                    (
                        self.last_scale_state,
                        self.last_scale_event,
                    ) = _extended_division_policy(
                        lr,
                        grad_norm,
                        self.singular_policy,
                        self.saturation_limit,
                    )
                    if grad_norm == 0.0:
                        continue
                    num = AbsoluteValue.from_float(lr)
                    den = AbsoluteValue.from_float(grad_norm)
                    scale = EternalRatio(
                        numerator=num, denominator=den
                    ).numerical_value()
                    update = scale * g

                    state = self.state.setdefault(p, {})
                    state.setdefault("momentum_buffer", None)
                    state.setdefault("step", 0)

                    if momentum > 0.0:
                        buf = state["momentum_buffer"]
                        if buf is None:
                            buf = update.clone()
                        else:
                            buf = momentum * buf + update
                        state["momentum_buffer"] = buf
                        update = buf
                    state["step"] += 1
                    p.data = p.data - update
            return loss

else:  # pragma: no cover - torch is optional
    if not TYPE_CHECKING:
        EternalTorchOptimizer = None


class AdaptiveEternalOptimizer:
    """Adam-like adaptive optimizer with ACT-stable accumulators.

    Implements bias-corrected first- and second-moment estimates plus a
    linear warmup and (optional) cosine decay schedule. Gradient clipping
    via ``max_grad_norm`` is performed using ACT-stable rescaling.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 0,
        total_steps: int = 0,
        singular_policy: SingularPolicy | str = SingularPolicy.PROPAGATE,
        saturation_limit: float = 1e12,
    ) -> None:
        _validate_lr(lr)
        _validate_weight_decay(weight_decay)
        _validate_max_grad_norm(max_grad_norm)
        _validate_warmup(warmup_steps)
        if total_steps < 0:
            raise ValueError(f"Invalid total_steps: {total_steps}")
        _validate_beta("beta1", betas[0])
        _validate_beta("beta2", betas[1])
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")

        normalized_params = self._normalize_param_groups(
            params, lr, betas, eps, weight_decay
        )
        self.param_groups: List[Dict[str, Any]] = normalized_params

        self.max_grad_norm = float(max_grad_norm)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.singular_policy = SingularPolicy(singular_policy)
        self.saturation_limit = float(saturation_limit)
        self.state: Dict[int, Dict[str, Any]] = {}
        self._global_step = 0
        self.last_clip_state: Optional[ExtendedRatio] = None
        self.last_clip_event: Optional[SingularArithmeticEvent] = None

    @staticmethod
    def _normalize_param_groups(
        params: Iterable[Any],
        lr: float,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
    ) -> List[Dict[str, Any]]:
        params = list(params)
        if not params:
            return [
                {
                    "params": [],
                    "lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay,
                }
            ]
        if isinstance(params[0], dict):
            groups: List[Dict[str, Any]] = []
            for g in params:
                g_lr = float(g.get("lr", lr))
                g_betas = tuple(g.get("betas", betas))
                g_eps = float(g.get("eps", eps))
                g_wd = float(g.get("weight_decay", weight_decay))
                if not math.isfinite(g_eps) or g_eps <= 0:
                    raise ValueError(f"Invalid eps: {g_eps}")
                _validate_lr(g_lr)
                _validate_weight_decay(g_wd)
                _validate_beta("beta1", g_betas[0])
                _validate_beta("beta2", g_betas[1])
                groups.append(
                    {
                        "params": list(g["params"]),
                        "lr": g_lr,
                        "betas": g_betas,
                        "eps": g_eps,
                        "weight_decay": g_wd,
                    }
                )
            return groups
        return [
            {
                "params": params,
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            }
        ]

    def _get_lr_scale(self, step: int) -> float:
        """Linear warmup followed by cosine decay (if ``total_steps`` set)."""
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return (step + 1) / float(self.warmup_steps)
        if self.total_steps > 0:
            decay_steps = max(1, self.total_steps - self.warmup_steps)
            progress = (step - self.warmup_steps) / float(decay_steps)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    def _clip_grad_by_norm(self, g: Any, max_norm: float) -> Any:
        if max_norm <= 0.0 or torch is None:
            return g
        grad_norm = float(torch.linalg.norm(g))
        if grad_norm <= max_norm:
            self.last_clip_state = ExtendedRatio.from_float(1.0)
            self.last_clip_event = None
            return g
        self.last_clip_state, self.last_clip_event = _extended_division_policy(
            max_norm,
            grad_norm,
            self.singular_policy,
            self.saturation_limit,
        )
        # ACT-normalised rescale: max_norm / grad_norm via EternalRatio
        num = AbsoluteValue.from_float(max_norm)
        den = AbsoluteValue.from_float(grad_norm)
        scale = EternalRatio(numerator=num, denominator=den).numerical_value()
        return scale * g

    def clip_scale_state(self, grad_norm: float) -> ExtendedRatio:
        """Return the ExtendedRatio state for ``max_grad_norm / grad_norm``."""
        return _extended_division_state(self.max_grad_norm, grad_norm)

    def clip_policy_state(
        self,
        grad_norm: float,
        policy: SingularPolicy | str | None = None,
    ) -> tuple[ExtendedRatio, Optional[SingularArithmeticEvent]]:
        """Return policy-resolved clipping scale and optional singular event."""
        return _extended_division_policy(
            self.max_grad_norm,
            grad_norm,
            policy or self.singular_policy,
            self.saturation_limit,
        )

    def _get_state(self, key: int) -> Dict[str, Any]:
        state = self.state.get(key)
        if state is None:
            state = {"step": 0, "exp_avg": None, "exp_avg_sq": None}
            self.state[key] = state
        return state

    def step(self, closure: Optional[Any] = None) -> Optional[Any]:
        loss = closure() if closure is not None else None
        if torch is None:
            return loss

        self._global_step += 1
        lr_scale = self._get_lr_scale(self._global_step - 1)

        for group in self.param_groups:
            lr = group["lr"] * lr_scale
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if weight_decay != 0.0:
                    g = g + weight_decay * p.data
                g = self._clip_grad_by_norm(g, self.max_grad_norm)

                state = self._get_state(id(p))
                state["step"] += 1
                t = state["step"]

                if state["exp_avg"] is None:
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                bias1 = 1 - beta1**t
                bias2 = 1 - beta2**t

                m_hat = exp_avg / bias1
                v_hat = exp_avg_sq / bias2

                denom = v_hat.sqrt().add_(eps)
                p.data = p.data - lr * (m_hat / denom)
        return loss
