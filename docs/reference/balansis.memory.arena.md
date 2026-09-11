# balansis.memory.arena

[Source](../../balansis/memory/arena.py) · [Reference index](index.md)

```python
class AbsoluteArena:
    def __init__(self) -> None:
        ...

    def alloc(self, magnitude: float, direction: int) -> AbsoluteValue:
        ...

    def size(self) -> int:
        ...
```
