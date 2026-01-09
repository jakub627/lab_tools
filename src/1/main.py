import collections
import copy
import random
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import TypeAlias, TypeGuard
from warnings import warn
import numpy as np
from ErrNum import ErrNum
from ErrNum2 import ErrNum as ErrNum2
from math import sqrt, isfinite


class NegativeStdDev(Exception):
    """Raise for a negative standard deviation"""

    pass


def is_expanded(lp: "LinearCombinationBase") -> TypeGuard["LinearCombinationExpanded"]:
    return isinstance(lp.linear_combo, dict)


def is_not_expanded(
    lp: "LinearCombinationBase",
) -> TypeGuard["LinearCombinationNonExpanded"]:
    return isinstance(lp.linear_combo, list)


class LinearCombinationBase:
    __slots__ = ("linear_combo",)

    def __init__(self, linear_combo):
        self.linear_combo = linear_combo

    def __bool__(self):
        return bool(self.linear_combo)

    def expanded(self):
        raise NotImplementedError

    def expand(self):
        raise NotImplementedError

    def __getstate__(self):
        return (self.linear_combo,)

    def __setstate__(self, state):
        (self.linear_combo,) = state


class LinearCombinationExpanded(LinearCombinationBase):
    def __init__(self, linear_combo: dict["Variable", float]):
        super().__init__(linear_combo)
        self.linear_combo: dict["Variable", float] = linear_combo

    def __getstate__(self) -> tuple[dict["Variable", float]]:
        return (self.linear_combo,)

    def __setstate__(self, state: tuple[dict["Variable", float]]):
        (self.linear_combo,) = state

    def expanded(self) -> bool:
        return True

    def expand(self) -> "LinearCombinationExpanded":
        return LinearCombinationExpanded(self.linear_combo)


class LinearCombinationNonExpanded(LinearCombinationBase):

    __slots__ = ("linear_combo",)

    def __init__(self, linear_combo: list[tuple[float, LinearCombinationBase]]):
        super().__init__(linear_combo)
        self.linear_combo: list[tuple[float, LinearCombinationBase]] = linear_combo

    def __getstate__(self) -> tuple[list[tuple[float, LinearCombinationBase]]]:
        return (self.linear_combo,)

    def __setstate__(self, state: tuple[list[tuple[float, LinearCombinationBase]]]):
        (self.linear_combo,) = state

    def expanded(self) -> bool:
        return False

    def expand(self) -> LinearCombinationExpanded:
        derivatives = collections.defaultdict(float)

        while self.linear_combo:
            (main_factor, main_expr) = self.linear_combo.pop()

            if is_expanded(main_expr):
                for var, factor in main_expr.linear_combo.items():
                    derivatives[var] += main_factor * factor
            elif is_not_expanded(main_expr):
                for factor, expr in main_expr.linear_combo:
                    self.linear_combo.append((main_factor * factor, expr))
        return LinearCombinationExpanded(derivatives)


class AffineScalarFunc(object):
    __slots__ = ("_nominal_value", "_linear_part")

    class dtype(object):
        type = staticmethod(lambda value: value)

    def __init__(self, nominal_value: float, linear_part: LinearCombinationBase):
        self._nominal_value = float(nominal_value)
        if not isinstance(linear_part, LinearCombinationBase):
            linear_part = LinearCombinationNonExpanded(linear_part)
        self._linear_part = linear_part

    @property
    def nominal_value(self) -> float:
        return self._nominal_value

    n = nominal_value

    @property
    def derivatives(self) -> dict["Variable", float]:

        if is_not_expanded(self._linear_part):
            self._linear_part = self._linear_part.expand()
            return self._linear_part.linear_combo
        elif is_expanded(self._linear_part):
            return self._linear_part.linear_combo
        else:
            raise NotImplementedError

    def error_components(self) -> dict["Variable", float]:
        error_components: dict["Variable", float] = {}

        for variable, derivative in self.derivatives.items():
            if variable._std_dev == 0:
                error_components[variable] = 0
            else:
                error_components[variable] = abs(derivative * variable._std_dev)

        return error_components

    @property
    def std_dev(self) -> float:
        return float(sqrt(sum(delta**2 for delta in self.error_components().values())))

    s = std_dev

    def __repr__(self):
        std_dev = self.std_dev
        if std_dev:
            std_dev_str = repr(std_dev)
        else:
            std_dev_str = "0"
        return f"{repr(self.nominal_value)}+/-{std_dev_str}"

    def __str__(self):
        return "uwu"

    def __deepcopy__(self, memo):
        return AffineScalarFunc(self._nominal_value, copy.deepcopy(self._linear_part))

    def __getstate__(self):
        all_attrs = {}
        try:
            all_attrs["__dict__"] = self.__dict__
        except AttributeError:
            pass
        all_slots = set()

        for cls in type(self).mro():
            slot_names = getattr(cls, "__slots__", ())
            if isinstance(slot_names, str):
                all_slots.add(slot_names)
            else:
                all_slots.update(slot_names)
        for name in all_slots:
            try:
                all_attrs[name] = getattr(self, name)
            except AttributeError:
                pass

        return all_attrs

    def __setstate__(self, data_dict):
        for name, value in data_dict.items():
            setattr(self, name, value)


class Variable(AffineScalarFunc):
    __slots__ = ("_std_dev", "tag")

    def __init__(self, value: float, std_dev: float, tag=None):
        value = float(value)
        super().__init__(value, LinearCombinationExpanded({self: 1.0}))
        self._std_dev = std_dev
        self.tag = tag

    @property
    def std_dev(self):
        return self._std_dev

    @std_dev.setter
    def std_dev(self, std_dev):
        if std_dev < 0 and isfinite(std_dev):
            raise NegativeStdDev("The standard deviation cannot be negative")
        self._std_dev = float(std_dev)

    def __repr__(self):
        num_repr = super().__repr__()

        if self.tag is None:
            return num_repr
        else:
            return f"< {self.tag} = {self.tag, num_repr} >"

    def __hash__(self):
        return id(self)

    def __copy__(self):
        return Variable(self.nominal_value, self.std_dev, self.tag)

    def __deepcopy__(self, memo):
        return self.__copy__()


def ufloat(nominal_value: float, std_dev: float = 0, tag=None):
    return Variable(nominal_value, std_dev, tag=tag)


def main():
    x = ufloat(10, 1)
    print(x)


if __name__ == "__main__":
    main()
