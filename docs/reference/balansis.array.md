# balansis.array

[Source](../../balansis/array.py) · [Reference index](index.md)

```python
def native_available() -> bool:
    ...
```

```python
def native_dot_available() -> bool:
    ...
```

```python
def native_gram_available() -> bool:
    ...
```

```python
def gram_pair(a: ArrayLike, b: ArrayLike, *, backend: Literal['auto', 'python', 'native']='auto') -> tuple[float, float, float]:
    ...
```

```python
def dot_array(a: ArrayLike, b: ArrayLike, *, backend: Literal['auto', 'python', 'native']='auto') -> float:
    ...
```

```python
def sum_array(values: ArrayLike, *, backend: Literal['auto', 'python', 'native']='auto') -> tuple[AbsoluteValue, float]:
    ...
```
