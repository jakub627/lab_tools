import random

import pytest
from src.ErrNum import ErrNum
import numpy as np


def test_attributes():
    v1 = 12, 0.21
    r1 = v1[1] / v1[0]

    p1 = ErrNum(v1[0], v1[1])
    p2 = ErrNum(0, 0.12)
    p3 = ErrNum(-1, 0.12)

    assert p1.v == v1[0], "Inputted values must match"
    assert p1.u == v1[1], "Inputted uncertainty must match"
    assert p1.r == r1, "Inputted uncertainty must match"

    assert p1.v == p1.value, "Values of aliased attributes must match"
    assert p1.u == p1.uncertainty, "Values of aliased attributes must match"
    assert p1.r == p1.relative, "Values of aliased attributes must match"

    assert p2.r == np.inf, "Relative error should be infinite"
    assert p3.r > 0, "Relative error should be positive"

    assert p1[0] == p1.v
    assert p1[1] == p1.u


def test_f_string_formatting():
    u = 0.21
    v = 12.25
    r = u / v
    p1 = ErrNum(v, u)

    assert f"{p1}" == f"{v}"
    assert f"{p1:.0f}" == f"{v:.0f}"
    assert f"{p1:U}" == f"{v} ± {2*u}"
    assert f"{p1:U1}" == f"{v} ± {u}"
    assert f"{p1:U2}" == f"{v} ± {2*u}"
    assert f"{p1:U3}" == f"{v} ± {3*u}"
    assert f"{p1:U4}" == f"{v} ± {4*u}"
    assert f"{ErrNum(3.1555,0):U3}" == "3.2 ± 0.0"

    assert f"{p1:R}" == f"{v} ± {2*r*100:.2g}%"
    assert f"{p1:R1}" == f"{v} ± {r*100:.2g}%"
    assert f"{p1:R2}" == f"{v} ± {2*r*100:.2g}%"
    assert f"{p1:R3}" == f"{v} ± {3*r*100:.2g}%"
    assert f"{p1:R4}" == f"{v} ± {4*r*100:.2g}%"

    p2 = ErrNum(0.0, 0.12)
    assert f"{p2:R}" == f"0.0 ± inf%"

    assert f"{ErrNum(3.1415,0.12):u}" == "3.14(12)"
    assert f"{ErrNum(3.1415,0.01):u}" == "3.142(10)"
    assert f"{ErrNum(314.15,1.2):u}" == "314.2(12)"
    assert f"{ErrNum(31415,15):u}" == "31415(15)"
    assert f"{ErrNum(3141515,152):u}" == "3141520(150)"
    assert f"{ErrNum(31415,0):u}" == "31000(0)"


def test_round_to_u():
    assert ErrNum(62626, 234).round_to_u() == ErrNum(62630, 230)
    assert ErrNum(62626, 2345).round_to_u() == ErrNum(62600, 2300)
    assert ErrNum(16161.5353, 6.161).round_to_u() == ErrNum(16161.5, 6.2)
    assert ErrNum(16161.53535, 0.0437).round_to_u() == ErrNum(16161.5354, 0.044)
    assert ErrNum(0, 0.0437).round_to_u() == ErrNum(0, 0.044)
    assert ErrNum(16161.53535, 0).round_to_u() == ErrNum(16000, 0)


def test_repr():
    v, u = 563.647, 4.892
    p1 = ErrNum(v, u)

    assert f"value={v:.4g}" in repr(p1)
    assert f"uncertainty={u:.4g}" in repr(p1)
    assert f"relative={u/v:.4g}" in repr(p1)
    assert f"{p1.__class__.__name__}" in repr(p1)


def test_to_ErrNum():
    assert ErrNum._to_ErrNum(1) == ErrNum(1, 0)
    assert ErrNum._to_ErrNum(2.1) == ErrNum(2.1, 0)
    assert ErrNum._to_ErrNum(1e3) == ErrNum(1e3, 0)
    assert ErrNum._to_ErrNum(ErrNum(123.421, 213.44)) == ErrNum(123.421, 213.44)

    with pytest.raises(TypeError, match="Cannot perform operation with str"):
        ErrNum._to_ErrNum("123")
    with pytest.raises(TypeError, match="Cannot perform operation with list"):
        ErrNum._to_ErrNum([1, 2])


def unc_add(x: ErrNum, y: ErrNum) -> ErrNum:
    unc = (x.u**2 + y.u**2) ** 0.5
    v = x.v + y.v
    return ErrNum(v, unc)


def test_add():
    p1 = ErrNum(11, 2)
    p2 = ErrNum(3, 0.2)
    p3 = ErrNum(1 / 3, 1 / 200)

    assert p1 + p2 == unc_add(p1, p2)
    assert p1 - p2 == unc_add(p1, -p2)
    assert p1 - p2 + p3 == unc_add(unc_add(p1, -p2), p3)

    assert -p1 == ErrNum(-11, 2)
    assert p1 + p2 == p2 + p1
    assert p1 - p2 == -p2 + p1
    assert +p1 == p1

    assert 2 * p1 == p1 + p1
    assert 3 * p1 == p1 + p1 + p1
    assert 3 * p1 == p1 + 2 * p1


def unc_mul(x: ErrNum, y: ErrNum) -> ErrNum:
    unc = ((x.u * y.v) ** 2 + (y.u * x.v) ** 2) ** 0.5
    v = x.v * y.v
    return ErrNum(v, unc)


def test_mul():
    p1 = ErrNum(11, 2)
    p2 = ErrNum(3, 0.2)
    p3 = ErrNum(1 / 3, 1 / 200)

    assert p1 * p2 == unc_mul(p1, p2)
    assert p1 * (-p2) == unc_mul(p1, -p2)
    assert (p1 * (-p2)) * p3 == unc_mul(unc_mul(p1, -p2), p3)

    assert p1 * p2 == p2 * p1
    assert p1 * (-p2) == -p2 * p1
