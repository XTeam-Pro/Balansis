import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.sep + "..")
from balansis.core.absolute import AbsoluteValue
from balansis.linalg.qr import qr_decompose
from balansis.linalg.svd import svd


def mat():
    return [
        [AbsoluteValue.from_float(1.0), AbsoluteValue.from_float(2.0)],
        [AbsoluteValue.from_float(3.0), AbsoluteValue.from_float(4.0)],
    ]


Q, R = qr_decompose(mat())
print([[x.to_float() for x in row] for row in Q])
print([[x.to_float() for x in row] for row in R])

try:
    U, S, Vt = svd(mat())
    print([[x.to_float() for x in row] for row in U])
    print([x.to_float() for x in S])
    print([[x.to_float() for x in row] for row in Vt])
except ImportError as e:
    print("svd optional:", str(e))
