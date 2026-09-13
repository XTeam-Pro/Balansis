# balansis.sets.resolver

[Source](../../balansis/sets/resolver.py) · [Reference index](index.md)

```python
def global_compensate(set_a: EternalSet, set_b: EternalSet) -> EternalSet:
    ...
```

```python
def verify_zero_sum(result_set: EternalSet, threshold: int=1000) -> list[AbsoluteValue]:
    ...
```

```python
def stream_compensate(iter1: Iterable[AbsoluteValue], iter2: Iterable[AbsoluteValue], limit: int | None=None) -> Iterator[AbsoluteValue]:
    ...
```

```python
def convergence_detector(result_iter: Iterable[AbsoluteValue], window: int=100, tol: float=1e-12) -> bool:
    ...
```
