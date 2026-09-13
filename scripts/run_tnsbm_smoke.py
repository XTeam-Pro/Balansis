import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.sep + "..")
from balansis.core.absolute import AbsoluteValue
from balansis.sets.eternal_set import EternalSet
from balansis.sets.generators import harmonic_generator
from balansis.sets.resolver import global_compensate, verify_zero_sum


def to_abs_list(lst):
    return [AbsoluteValue.from_float(float(x)) for x in lst]


set_a = EternalSet(to_abs_list([1, 2, 3]))
set_b = EternalSet(to_abs_list([-1, -2, -3]))
res1 = global_compensate(set_a, set_b)
print(list(res1))

h_pos = EternalSet(harmonic_generator(+1), is_infinite=True, rule_name="harmonic+")
h_neg = EternalSet(harmonic_generator(-1), is_infinite=True, rule_name="harmonic-")
res2 = global_compensate(h_pos, h_neg)
print(len(verify_zero_sum(res2, threshold=1000)))

set_c = EternalSet(to_abs_list([10, 20]))
set_d = EternalSet(to_abs_list([-10]))
res3 = global_compensate(set_c, set_d)
print([x.to_float() for x in res3])
