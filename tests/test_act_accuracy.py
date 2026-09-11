# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.
#
# This file is part of Balansis (dual-licensed AGPLv3 / commercial).
"""Accuracy tests that lock the whitepaper's numerical-stability claims.

These assert the *measured* behaviour behind ACT_WHITEPAPER_v1.md: compensated
summation (S 5.2/5.3), correctly-rounded dot products (S 4.6), and the genuine
ACT-compensated one-sided Jacobi SVD (S 4.3). They fail if a change silently
regresses the accuracy the whitepaper promises.
"""

import math
from fractions import Fraction

import numpy as np
import pytest

from balansis.core._eft import dot2, two_product, two_sum
from balansis.core.absolute import AbsoluteValue
from balansis.core.operations import Operations
from balansis.linalg.svd import svd
from balansis.numpy_integration import compensated_dot_product


def _relerr(approx: float, exact) -> float:
    exact = float(exact)
    return abs(approx - exact) / abs(exact) if exact != 0 else abs(approx)


def test_two_sum_is_exact():
    a, b = 1e16, 3.0
    s, e = two_sum(a, b)
    assert Fraction(s) + Fraction(e) == Fraction(a) + Fraction(b)


def test_two_product_is_exact():
    a, b = 1.0000000001e8, 9.9999999e7
    p, e = two_product(a, b)
    assert Fraction(p) + Fraction(e) == Fraction(a) * Fraction(b)


def test_dot2_correctly_rounded_on_ill_conditioned():
    """Ogita-Rump Dot2 recovers full accuracy where naive np.dot loses everything."""
    rng = np.random.default_rng(7)
    n = 40
    x = rng.standard_normal(n) * 10.0 ** rng.integers(6, 12, n)
    y = rng.standard_normal(n) * 10.0 ** rng.integers(6, 12, n)
    partial = sum(Fraction(float(x[i])) * Fraction(float(y[i])) for i in range(n - 1))
    y[n - 1] = float((Fraction(1) - partial) / Fraction(float(x[n - 1])))
    true = sum(Fraction(float(x[i])) * Fraction(float(y[i])) for i in range(n))
    # condition number of this dot product is ~1e17
    assert _relerr(float(np.dot(x, y)), true) > 1e-2  # naive fails
    assert _relerr(dot2(x, y), true) < 1e-14  # Dot2 is essentially exact
    assert _relerr(compensated_dot_product(x, y), true) < 1e-14


def test_sequence_sum_beats_classic_kahan_on_cancellation():
    """S 5.2: ACT summation recovers [big, small, -big] where classic Kahan fails."""
    vals = []
    for _ in range(200):
        vals += [1e16, 1.0, -1e16]
    true = 200.0

    def classic_kahan(v):
        s = c = 0.0
        for x in v:
            y = x - c
            t = s + y
            c = (t - s) - y
            s = t
        return s

    naive = 0.0
    for v in vals:
        naive += v
    assert naive == 0.0  # float64 loses everything
    assert classic_kahan(vals) == 0.0  # classic Kahan also fails on this pattern
    act, _ = Operations.sequence_sum([AbsoluteValue.from_float(v) for v in vals])
    assert _relerr(act.to_float(), true) < 1e-13  # ACT (Neumaier) recovers it


def test_act_jacobi_svd_reconstruction_and_orthogonality():
    rng = np.random.default_rng(3)
    for shape in [(6, 6), (8, 3), (3, 7)]:
        M = rng.standard_normal(shape)
        A = [[AbsoluteValue.from_float(float(v)) for v in row] for row in M]
        res = svd(A, method="act_jacobi")
        U = np.array([[v.to_float() for v in r] for r in res.U])
        S = np.array([v.to_float() for v in res.S])
        Vt = np.array([[v.to_float() for v in r] for r in res.Vt])
        assert np.linalg.norm(M - U @ np.diag(S) @ Vt) < 1e-12
        assert np.linalg.norm(U.T @ U - np.eye(U.shape[1])) < 1e-12
        s_numpy = np.linalg.svd(M, compute_uv=False)
        s_act = np.sort(S)[::-1][: len(s_numpy)]
        assert np.max(np.abs(s_act - s_numpy)) < 1e-10


def test_act_jacobi_svd_high_relative_accuracy_ill_conditioned():
    """S 4.3: genuine ACT SVD keeps relative accuracy on tiny singular values."""
    rng = np.random.default_rng(11)
    D = np.diag([1.0, 1e-5, 1e-10, 1e-15])
    Q1, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    Q2, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    M = Q1 @ D @ Q2.T
    A = [[AbsoluteValue.from_float(float(v)) for v in row] for row in M]
    res = svd(A, method="act_jacobi")
    U = np.array([[v.to_float() for v in r] for r in res.U])
    S = np.sort(np.array([v.to_float() for v in res.S]))[::-1]
    Vt = np.array([[v.to_float() for v in r] for r in res.Vt])
    assert np.linalg.norm(M - U @ np.diag(np.sort(S)[::-1]) @ Vt) < 1e-12
    # smallest singular value recovered to good relative accuracy
    assert abs(S[3] - 1e-15) / 1e-15 < 0.1


def test_svd_rejects_unknown_method():
    A = [[AbsoluteValue.from_float(1.0)]]
    with pytest.raises(ValueError):
        svd(A, method="does_not_exist")
