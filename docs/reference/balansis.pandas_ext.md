# balansis.pandas_ext

[Source](../../balansis/pandas_ext.py) · [Reference index](index.md)

```python
class AbsoluteValueDtype(ExtensionDtype):
    @classmethod
    def construct_array_type(cls) -> Type[AbsoluteArray]:
        ...
```

```python
class AbsoluteArray(ExtensionArray):
    def __init__(self, values: Any) -> None:
        ...

    @property
    def dtype(self) -> AbsoluteValueDtype:
        ...

    @property
    def nbytes(self) -> int:
        ...

    def isna(self) -> np.ndarray:
        ...

    def take(self, indices: Any, allow_fill: bool=False, fill_value: Any=None) -> AbsoluteArray:
        ...

    def copy(self) -> AbsoluteArray:
        ...

    def to_numpy(self, dtype: Any=None, copy: bool=False, na_value: Any=None) -> np.ndarray:
        ...

    def astype(self, dtype: Any, copy: bool=True) -> Any:
        ...
```
