"""ACT-aware optimizers for machine learning.

This module provides optimizers that use EternalRatio-based learning rate
scaling with optional momentum and weight decay, all implemented using
ACT-compensated arithmetic for enhanced numerical stability.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    from torch import Tensor
except ImportError:
    torch = None  # type: ignore[assignment]
    Tensor = None  # type: ignore[assignment, misc]

from balansis.core.absolute import AbsoluteValue
from balansis.core.eternity import EternalRatio
from balansis.core.operations import Operations

__all__: List[str] = ["EternalOptimizer", "AdaptiveEternalOptimizer"]


class EternalOptimizer:
    """ACT-aware optimizer with momentum and weight decay support.

    Uses EternalRatio-based learning rate scaling to maintain numerical
    stability during gradient descent. Supports exponential moving average
    momentum and ACT-compensated weight decay.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default: 1e-3).
        momentum: Momentum factor for exponential moving average (default: 0.0).
        weight_decay: Weight decay coefficient with ACT compensation (default: 0.0).
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        """Initialize the EternalOptimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            momentum: Momentum factor (0.0 means no momentum).
            weight_decay: Weight decay coefficient (0.0 means no decay).

        Raises:
            ValueError: If lr, momentum, or weight_decay are invalid.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.params: List[Any] = list(params)
        self.lr: float = lr
        self.momentum: float = momentum
        self.weight_decay: float = weight_decay
        self.state: Dict[int, Dict[str, Any]] = {}

    def _get_state(self, param_id: int) -> Dict[str, Any]:
        """Get or initialize state for a parameter.

        Args:
            param_id: Unique identifier for the parameter.

        Returns:
            State dictionary containing momentum buffer and step count.
        """
        if param_id not in self.state:
            self.state[param_id] = {"momentum_buffer": None, "step": 0}
        return self.state[param_id]

    def step(self) -> None:
        """Perform a single optimization step.

        Updates all parameters using ACT-compensated gradient descent
        with optional momentum and weight decay.
        """
        if torch is None:
            return

        for p in self.params:
            if not hasattr(p, "grad") or p.grad is None:
                continue

            g = p.grad
            param_id = id(p)
            state = self._get_state(param_id)
            state["step"] += 1

            # Apply weight decay with ACT compensation
            if self.weight_decay != 0.0:
                decay_factor = 1.0 - self.lr * self.weight_decay
                decay_av = AbsoluteValue.from_float(
                    max(decay_factor, Operations.COMPENSATION_THRESHOLD)
                )
                p.data = p.data * decay_av.to_float()

            # Apply momentum (exponential moving average)
            if self.momentum != 0.0:
                if state["momentum_buffer"] is None:
                    state["momentum_buffer"] = torch.clone(g).detach()
                else:
                    state["momentum_buffer"].mul_(self.momentum).add_(
                        g, alpha=1.0 - self.momentum
                    )
                g = state["momentum_buffer"]

            # Compute ACT-scaled learning rate via EternalRatio
            grad_norm = float(torch.linalg.norm(g))
            num = AbsoluteValue.from_float(self.lr)

            if grad_norm > 0:
                den = AbsoluteValue.from_float(grad_norm)
            else:
                den = AbsoluteValue.unit_positive()

            ratio = EternalRatio(numerator=num, denominator=den)
            scaled_lr = ratio.numerical_value()

            # Update parameters
            p.data = p.data - scaled_lr * g


class AdaptiveEternalOptimizer:
    """Adam-like optimizer with ACT-compensated moment tracking.

    Uses EternalRatio for first and second moment bias correction,
    ACT-compensated gradient clipping, and warmup + cosine decay
    learning rate schedule. Supports parameter groups with different
    hyperparameters.

    Args:
        params: Iterable of parameters or list of parameter group dicts.
            Each group dict must have a "params" key and may override
            lr, betas, eps, weight_decay, max_grad_norm.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for computing running averages of gradient
            and its square (default: (0.9, 0.999)).
        eps: Term added to denominator for stability (default: 1e-8).
        weight_decay: Weight decay coefficient (default: 0.0).
        max_grad_norm: Maximum gradient norm for ACT clipping.
            Set to 0 to disable clipping (default: 1.0).
        warmup_steps: Number of linear warmup steps (default: 0).
        total_steps: Total steps for cosine decay. 0 means no decay (default: 0).
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 0,
        total_steps: int = 0,
    ) -> None:
        """Initialize AdaptiveEternalOptimizer.

        Args:
            params: Iterable of parameters or list of parameter group dicts.
            lr: Learning rate.
            betas: Coefficients for moment EMAs.
            eps: Numerical stability term.
            weight_decay: Weight decay coefficient.
            max_grad_norm: Max gradient norm for clipping (0 disables).
            warmup_steps: Linear warmup step count.
            total_steps: Total steps for cosine decay schedule.

        Raises:
            ValueError: If any hyperparameter is invalid.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if max_grad_norm < 0.0:
            raise ValueError(f"Invalid max_grad_norm value: {max_grad_norm}")
        if warmup_steps < 0:
            raise ValueError(f"Invalid warmup_steps value: {warmup_steps}")

        # Support parameter groups (list of dicts) or flat param list
        if isinstance(params, list) and len(params) > 0 and isinstance(params[0], dict):
            self.param_groups: List[Dict[str, Any]] = []
            for group in params:
                self.param_groups.append({
                    "params": list(group["params"]),
                    "lr": group.get("lr", lr),
                    "betas": group.get("betas", betas),
                    "eps": group.get("eps", eps),
                    "weight_decay": group.get("weight_decay", weight_decay),
                    "max_grad_norm": group.get("max_grad_norm", max_grad_norm),
                })
        else:
            self.param_groups = [{
                "params": list(params),
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "max_grad_norm": max_grad_norm,
            }]

        self.warmup_steps: int = warmup_steps
        self.total_steps: int = total_steps
        self.state: Dict[int, Dict[str, Any]] = {}
        self._global_step: int = 0

    def _get_lr_scale(self, step: int) -> float:
        """Compute learning rate scale with warmup and cosine decay.

        Uses EternalRatio for stable ratio computation.

        Args:
            step: Current step (0-indexed).

        Returns:
            Scale factor for the base learning rate.
        """
        if self.warmup_steps > 0 and step < self.warmup_steps:
            num = AbsoluteValue.from_float(float(step + 1))
            den = AbsoluteValue.from_float(float(self.warmup_steps))
            ratio = EternalRatio(numerator=num, denominator=den)
            return ratio.numerical_value()

        if self.total_steps > 0:
            decay_steps = max(1, self.total_steps - self.warmup_steps)
            elapsed = float(min(step - self.warmup_steps, decay_steps))
            progress_num = AbsoluteValue.from_float(elapsed)
            progress_den = AbsoluteValue.from_float(float(decay_steps))
            progress = EternalRatio(
                numerator=progress_num, denominator=progress_den
            ).numerical_value()
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return 1.0

    def _clip_grad_by_norm(self, g: Any, max_norm: float) -> Any:
        """Clip gradient via ACT normalization.

        Computes gradient norm using compensated sum and scales the gradient
        if norm exceeds the threshold (soft scaling, no hard clip).

        Args:
            g: Gradient tensor.
            max_norm: Maximum gradient norm. 0 disables clipping.

        Returns:
            Possibly scaled gradient tensor.
        """
        if max_norm <= 0.0:
            return g

        grad_values = g.flatten().tolist()
        sq_values = [
            AbsoluteValue.from_float(v * v) for v in grad_values
        ]
        sq_sum, _ = Operations.sequence_sum(sq_values)
        grad_norm = math.sqrt(max(sq_sum.to_float(), 0.0))

        if grad_norm > max_norm:
            num = AbsoluteValue.from_float(max_norm)
            den = AbsoluteValue.from_float(grad_norm)
            scale = EternalRatio(
                numerator=num, denominator=den
            ).numerical_value()
            return g * scale

        return g

    def step(self) -> None:
        """Perform a single optimization step.

        Updates all parameters using Adam-like updates with ACT-compensated
        bias correction, gradient clipping, and learning rate scheduling.
        """
        if torch is None:
            return

        self._global_step += 1
        lr_scale = self._get_lr_scale(self._global_step - 1)

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            max_norm = group["max_grad_norm"]

            for p in group["params"]:
                if not hasattr(p, "grad") or p.grad is None:
                    continue

                g = p.grad
                param_id = id(p)

                if param_id not in self.state:
                    self.state[param_id] = {
                        "step": 0,
                        "m1": torch.zeros_like(p.data),
                        "m2": torch.zeros_like(p.data),
                    }

                state = self.state[param_id]
                state["step"] += 1
                t = state["step"]

                # Weight decay with ACT compensation
                if wd != 0.0:
                    decay_val = max(
                        1.0 - lr * wd, Operations.COMPENSATION_THRESHOLD
                    )
                    decay_av = AbsoluteValue.from_float(decay_val)
                    p.data = p.data * decay_av.to_float()

                # Gradient clipping via ACT normalization
                g = self._clip_grad_by_norm(g, max_norm)

                # Update first moment (mean) via EternalRatio-scaled coefficients
                m1 = state["m1"]
                beta1_er = EternalRatio.from_float(beta1)
                m1.mul_(beta1_er.numerical_value()).add_(
                    g, alpha=1.0 - beta1_er.numerical_value()
                )

                # Update second moment (variance) via EternalRatio-scaled coefficients
                m2 = state["m2"]
                beta2_er = EternalRatio.from_float(beta2)
                m2.mul_(beta2_er.numerical_value()).add_(
                    g * g, alpha=1.0 - beta2_er.numerical_value()
                )

                # Bias correction using ACT-compensated division
                bias1_val = max(
                    1.0 - beta1 ** t, Operations.COMPENSATION_THRESHOLD
                )
                bias1_ratio = EternalRatio(
                    numerator=AbsoluteValue.from_float(1.0),
                    denominator=AbsoluteValue.from_float(bias1_val),
                )
                m1_hat = m1 * bias1_ratio.numerical_value()

                bias2_val = max(
                    1.0 - beta2 ** t, Operations.COMPENSATION_THRESHOLD
                )
                bias2_ratio = EternalRatio(
                    numerator=AbsoluteValue.from_float(1.0),
                    denominator=AbsoluteValue.from_float(bias2_val),
                )
                m2_hat = m2 * bias2_ratio.numerical_value()

                # Adam update: p -= lr * lr_scale * m1_hat / (sqrt(m2_hat) + eps)
                denom = torch.sqrt(m2_hat) + eps
                update = m1_hat / denom
                p.data = p.data - (lr * lr_scale) * update


if torch is not None:
    __all__.append("EternalTorchOptimizer")

    class EternalTorchOptimizer(torch.optim.Optimizer):
        """PyTorch-compatible ACT optimizer with momentum and weight decay.

        Integrates with PyTorch's optimizer interface while using EternalRatio-
        based learning rate scaling with ACT-compensated weight decay.

        Args:
            params: Iterable of parameters or dicts defining parameter groups.
            lr: Learning rate (default: 1e-3).
            momentum: Momentum factor (default: 0.0).
            weight_decay: Weight decay coefficient (default: 0.0).
        """

        def __init__(
            self,
            params: Any,
            lr: float = 1e-3,
            momentum: float = 0.0,
            weight_decay: float = 0.0,
        ) -> None:
            """Initialize EternalTorchOptimizer.

            Args:
                params: Parameters to optimize.
                lr: Learning rate.
                momentum: Momentum factor.
                weight_decay: Weight decay coefficient.
            """
            defaults: Dict[str, Any] = dict(
                lr=lr, momentum=momentum, weight_decay=weight_decay
            )
            super().__init__(params, defaults)

        @torch.no_grad()  # type: ignore[misc]
        def step(self, closure: Optional[Any] = None) -> Optional[Any]:
            """Perform a single optimization step.

            Args:
                closure: A closure that reevaluates the model and returns loss.

            Returns:
                Optional loss value from closure.
            """
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                lr = group["lr"]
                momentum = group.get("momentum", 0.0)
                weight_decay = group.get("weight_decay", 0.0)

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    g = p.grad

                    # Get parameter state
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["momentum_buffer"] = None
                    state["step"] += 1

                    # Apply weight decay with ACT compensation
                    if weight_decay != 0.0:
                        decay_factor = 1.0 - lr * weight_decay
                        decay_av = AbsoluteValue.from_float(
                            max(decay_factor, 1e-15)
                        )
                        p.data.mul_(decay_av.to_float())

                    # Apply momentum
                    if momentum != 0.0:
                        if state["momentum_buffer"] is None:
                            state["momentum_buffer"] = torch.clone(g).detach()
                        else:
                            state["momentum_buffer"].mul_(momentum).add_(
                                g, alpha=1.0 - momentum
                            )
                        g = state["momentum_buffer"]

                    # Compute ACT-scaled learning rate
                    grad_norm = float(torch.linalg.norm(g))
                    num = AbsoluteValue.from_float(lr)
                    den = (
                        AbsoluteValue.from_float(grad_norm)
                        if grad_norm > 0
                        else AbsoluteValue.unit_positive()
                    )
                    ratio = EternalRatio(numerator=num, denominator=den)

                    p.data.add_(g, alpha=-ratio.numerical_value())

            return loss
