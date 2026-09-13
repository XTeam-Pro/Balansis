from balansis.core.absolute import AbsoluteValue
from balansis.sets.eternal_set import EternalSet
from balansis.sets.generators import harmonic_generator
from balansis.sets.resolver import global_compensate, verify_zero_sum


def to_abs_list(lst):
    return [AbsoluteValue.from_float(float(x)) for x in lst]


def test_full_annihilation():
    set_a = EternalSet(to_abs_list([1, 2, 3]))
    set_b = EternalSet(to_abs_list([-1, -2, -3]))
    res = global_compensate(set_a, set_b)
    residuals = list(res)
    assert residuals == []


def test_harmonic_compensation():
    h_pos = EternalSet(harmonic_generator(+1), is_infinite=True, rule_name="harmonic+")
    h_neg = EternalSet(harmonic_generator(-1), is_infinite=True, rule_name="harmonic-")
    res = global_compensate(h_pos, h_neg)
    tail = verify_zero_sum(res, threshold=1000)
    assert len(tail) == 0


def test_partial_compensation():
    set_a = EternalSet(to_abs_list([10, 20]))
    set_b = EternalSet(to_abs_list([-10]))
    res = global_compensate(set_a, set_b)
    vals = [x.to_float() for x in res]
    assert vals == [20.0]
