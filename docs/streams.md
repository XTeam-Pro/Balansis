# Streams and sets

`EternalSet(source, is_infinite=False, rule_name="custom")` wraps an iterable of
`AbsoluteValue`. Iteration consumes the supplied iterable; generators retain
their usual single-pass behavior.

`global_compensate(a, b)` adds corresponding terms, using zero after a finite
input ends. Finite results omit zero terms. A result marked infinite emits one
term for each pair, including zeros.

```python
from itertools import islice
from balansis import EternalSet, harmonic_generator, global_compensate

positive = EternalSet(harmonic_generator(1), is_infinite=True)
negative = EternalSet(harmonic_generator(-1), is_infinite=True)
assert all(v.is_absolute() for v in islice(global_compensate(positive, negative), 10))
```

`grandis_generator()` alternates `+1` and `-1`. `verify_zero_sum` inspects a
bounded number of terms and returns nonzero residuals. `stream_compensate`
accepts a pair-count limit; `convergence_detector` looks for a window of small
terms. [Source](../balansis/sets).
