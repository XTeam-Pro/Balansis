import os
import random
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.sep + "..")
from balansis.core.absolute import AbsoluteValue


def gen_value():
    m = random.choice([0.0, 1.0, random.uniform(0.0, 10.0)])
    d = random.choice([1, -1])
    return AbsoluteValue(magnitude=m, direction=d)


def check_compensation():
    x = gen_value()
    y = AbsoluteValue(magnitude=x.magnitude, direction=-x.direction)
    return (x + y).is_absolute()


def check_commutativity():
    a = gen_value()
    b = gen_value()
    return (a + b) == (b + a)


def check_associativity():
    a = gen_value()
    b = gen_value()
    c = gen_value()
    return ((a + b) + c) == (a + (b + c))


def main(n=10000):
    comp_ok = 0
    comm_ok = 0
    assoc_ok = 0
    assoc_violations = 0
    for _ in range(n):
        comp_ok += 1 if check_compensation() else 0
        comm_ok += 1 if check_commutativity() else 0
        if check_associativity():
            assoc_ok += 1
        else:
            assoc_violations += 1
    print("compensation", comp_ok, "/", n)
    print("commutativity", comm_ok, "/", n)
    print("associativity_ok", assoc_ok, "violations", assoc_violations)


if __name__ == "__main__":
    main()
