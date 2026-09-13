# balansis.ml.optimizer

[Source](../../balansis/ml/optimizer.py) · [Reference index](index.md)

```python
class EternalOptimizer:
    def __init__(self, params: Iterable[Any], lr: float=0.001, momentum: float=0.0, weight_decay: float=0.0, singular_policy: SingularPolicy | str=SingularPolicy.PROPAGATE, saturation_limit: float=1000000000000.0) -> None:
        ...

    def scale_state(self, grad_norm: float) -> ExtendedRatio:
        ...

    def scale_policy_state(self, grad_norm: float, policy: SingularPolicy | str | None=None) -> tuple[ExtendedRatio, Optional[SingularArithmeticEvent]]:
        ...

    def step(self, closure: Optional[Any]=None) -> Optional[Any]:
        ...
```

```python
class EternalTorchOptimizer(torch.optim.Optimizer):
    def __init__(self, params: Iterable[Any], lr: float=0.001, momentum: float=0.0, weight_decay: float=0.0, singular_policy: SingularPolicy | str=SingularPolicy.PROPAGATE, saturation_limit: float=1000000000000.0) -> None:
        ...

    def step(self, closure: Optional[Callable[[], Any]]=None) -> Any:
        ...
```

```python
class AdaptiveEternalOptimizer:
    def __init__(self, params: Iterable[Any], lr: float=0.001, betas: Tuple[float, float]=(0.9, 0.999), eps: float=1e-08, weight_decay: float=0.0, max_grad_norm: float=1.0, warmup_steps: int=0, total_steps: int=0, singular_policy: SingularPolicy | str=SingularPolicy.PROPAGATE, saturation_limit: float=1000000000000.0) -> None:
        ...

    def clip_scale_state(self, grad_norm: float) -> ExtendedRatio:
        ...

    def clip_policy_state(self, grad_norm: float, policy: SingularPolicy | str | None=None) -> tuple[ExtendedRatio, Optional[SingularArithmeticEvent]]:
        ...

    def step(self, closure: Optional[Any]=None) -> Optional[Any]:
        ...
```
