from collections.abc import Callable
from decimal import ROUND_HALF_UP, Decimal
import re
import numpy as np
from typing import Any

import sympy as sp


class ErrNum:
    def __init__(self, value: float, uncertainty: float = 0) -> None:
        self.value = value
        self.uncertainty = uncertainty

    @property
    def relative(self) -> float:
        if self.value == 0:
            return np.inf
        return abs(self.uncertainty / self.value)

    @property
    def r(self) -> float:
        return self.relative

    @property
    def v(self) -> float:
        return self.value

    @property
    def u(self) -> float:
        return self.uncertainty

    def __format__(self, format_spec: str) -> str:

        m: Callable[[str], re.Match[str] | None] = lambda s: re.fullmatch(
            rf"{s}([1-9]\d*)?", format_spec
        )
        if mU := m("U"):  # extended uncertainty - "v ± k*u"
            p = self.round_to_u()
            k = int(mU.group(1)) if mU.group(1) is not None else 2
            return f"{p.v} ± {k * p.u}"

        if mR := m("R"):  # extended uncertainty - "v ± k*r%"
            p = self.round_to_u()
            k = int(mR.group(1)) if mR.group(1) is not None else 2
            return f"{p.v} ± {k*p.r*100:.2g}%"

        if format_spec == "u":  # extended uncertainty - "v(u)"
            rounded = self.round_to_u()
            digits = max(
                0, -int(np.floor(np.log10(abs(rounded.u)))) + 1 if rounded.u > 0 else 0
            )

            u_str = re.sub(r"0*\.0*", "", f"{rounded.u:.{digits}f}")
            return f"{rounded.v:.{digits}f}({u_str})"

        return format(self.value, format_spec)

    def __getitem__(self, index) -> float:
        return [self.value, self.uncertainty][index]

    def round_to_u(self) -> "ErrNum":
        base = self.u if self.u else self.v
        exponent = int(np.floor(np.log10(abs(base)))) - 1
        decimal_places = -exponent

        rounded_u = float(
            Decimal(str(self.u)).quantize(
                Decimal(f"1e{-decimal_places}"), rounding=ROUND_HALF_UP
            )
        )
        rounded_v = float(
            Decimal(str(self.v)).quantize(
                Decimal(f"1e{-decimal_places}"), rounding=ROUND_HALF_UP
            )
        )
        return ErrNum(rounded_v, rounded_u)

    def __str__(self) -> str:
        return f"{self:U}"

    def __repr__(self) -> str:
        f = ".4g"
        txt = ", ".join(
            [
                f"value={self.value:{f}}",
                f"uncertainty={self.uncertainty:{f}}",
                f"relative={self.relative:{f}}",
            ]
        )
        return f"{self.__class__.__name__}({txt})"

    @staticmethod
    def _to_ErrNum(other: Any) -> "ErrNum":
        if isinstance(other, ErrNum):
            return other
        elif isinstance(other, (float, int)):
            return ErrNum(other, 0.0)
        else:
            raise TypeError(f"Cannot perform operation with {other.__class__.__name__}")

    def __add__(
        self,
        other: "float | int | ErrNum",
    ):
        other = self._to_ErrNum(other)
        new_value = self.v + other.v
        new_unc = self._calc_unc(other, "x+y")
        return ErrNum(new_value, new_unc)

    __radd__ = __add__

    def __neg__(self) -> "ErrNum":
        return ErrNum(-self.value, self.uncertainty)

    def __pos__(self):
        return self

    def __sub__(self, other: "float | int | ErrNum"):
        return self.__add__(-other)

    def __rsub__(self, other: "float | int | ErrNum"):
        return -self.__add__(other)

    def __mul__(self, other: "float | int | ErrNum"):
        other = self._to_ErrNum(other)
        new_value = self.value * other.value
        new_unc = self._calc_unc(other, "x*y")
        return ErrNum(new_value, new_unc)

    __rmul__ = __mul__

    def __truediv__(self, other: "float | int | ErrNum"):
        other = self._to_ErrNum(other)
        new_value = self.value / other.value
        new_unc = self._calc_unc(other, "x/y")
        return ErrNum(new_value, new_unc)

    def __rtruediv__(self, other: "float | int | ErrNum"):
        other = self._to_ErrNum(other)
        return other.__truediv__(self)

    def equal_extended_uncertainty(
        self, other: "float | int | ErrNum", k: int = 2
    ) -> bool:
        other = self._to_ErrNum(other)
        U_diff: float = k * np.sqrt(self.u**2 + other.u**2)
        diff: float = np.abs(self.v - other.v)
        return bool(diff <= U_diff)

    def _calc_unc(self, other: "float | int | ErrNum", expr: str) -> float:
        other = self._to_ErrNum(other)
        x, y, ux, uy = sp.symbols("x,y,u_x,u_y")
        f = sp.sympify(expr)
        df_dx = sp.diff(f, x)
        df_dy = sp.diff(f, y)

        if self is other:
            delta_expr = ((sp.diff(expr, x) * ux) ** 2) ** 0.5 + (
                (sp.diff(expr, y) * uy) ** 2
            ) ** 0.5
        else:
            delta_expr = (
                (sp.diff(expr, x) * ux) ** 2 + (sp.diff(expr, y) * uy) ** 2
            ) ** 0.5

        delta_val = delta_expr.subs(
            {"x": self.v, "y": other.v, "u_x": self.u, "u_y": other.u}
        )
        return float(delta_val)

    def __pow__(self, other: "float | int | ErrNum") -> "ErrNum":
        other = self._to_ErrNum(other)
        new_value = self.v**other.v

        new_unc = self._calc_unc(other, "x**y")
        return ErrNum(new_value, new_unc)

    # --- (without uncertainty) ---
    def __eq__(self, other: object):
        o = self._to_ErrNum(other)
        if isinstance(other, ErrNum):
            return bool(np.isclose(self.v, o.v) and np.isclose(self.u, o.u))
        else:
            return self.value == o.value

    def __lt__(self, other: object):
        other = self._to_ErrNum(other)
        return self.value < other.value

    def __le__(self, other: object):
        other = self._to_ErrNum(other)
        return self.value <= other.value

    def __gt__(self, other: object):
        other = self._to_ErrNum(other)
        return self.value > other.value

    def __ge__(self, other: object):
        other = self._to_ErrNum(other)
        return self.value >= other.value
