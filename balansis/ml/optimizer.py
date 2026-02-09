"""ACT-aware optimizers for machine learning.

This module provides optimizers that use EternalRatio-based learning rate
scaling with optional momentum and weight decay, all implemented using
ACT-compensated arithmetic for enhanced numerical stability.
"""

from typing import Any, Dict, List, Optional

try:
    import torch
    from torch import Tensor
except ImportError:
    torch = None  # type: ignore[assignment]
    Tensor = None  # type: ignore[assignment, misc]

from balansis.core.absolute import AbsoluteValue
from balansis.core.eternity import EternalRatio
from balansis.core.operations import Operations


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


if torch is not None:

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
