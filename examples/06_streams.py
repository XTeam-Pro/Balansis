from itertools import islice
from balansis import EternalSet, harmonic_generator, global_compensate

positive = EternalSet(harmonic_generator(1), is_infinite=True)
negative = EternalSet(harmonic_generator(-1), is_infinite=True)
assert all(v.is_absolute() for v in islice(global_compensate(positive, negative), 10))
