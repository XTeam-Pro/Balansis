import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.sep + "..")
from balansis.core.absolute import AbsoluteValue
from balansis.core.eternity import EternalRatio
from balansis.linalg.gemm import matmul


def main():
    a = AbsoluteValue.from_float(2.0)
    print(a.log())
    print(a.exp().to_float())
    print(a.sin())
    print(a.cos())
    r = EternalRatio.from_values(6.0, 2.0)
    print(r.log())
    print(r.exp().numerical_value())
    print(r.sin())
    m = [
        [AbsoluteValue.from_float(1.0), AbsoluteValue.from_float(2.0)],
        [AbsoluteValue.from_float(3.0), AbsoluteValue.from_float(4.0)],
    ]
    n = [
        [AbsoluteValue.from_float(5.0), AbsoluteValue.from_float(6.0)],
        [AbsoluteValue.from_float(7.0), AbsoluteValue.from_float(8.0)],
    ]
    res, _ = matmul(m, n)
    for row in res:
        print([x.to_float() for x in row])


if __name__ == "__main__":
    main()
