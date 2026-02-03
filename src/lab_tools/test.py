from __future__ import annotations
from collections import defaultdict
import copy
from typing import TypeGuard, cast

import numpy as np

from .formatting import format_float


class NegativeStdDev(Exception):
    """Raise for a negative standard deviation"""

    pass


ExpandedLinearComboType = dict["Variable", float]
NonExpandedLinearComboType = list[tuple[float, "LinearCombination"]]
LinearComboType = ExpandedLinearComboType | NonExpandedLinearComboType


class LinearCombination:
    __slots__ = "linear_combo"

    def __bool__(self):
        return bool(self.linear_combo)

    def __init__(self, linear_combo: LinearComboType) -> None:
        self.linear_combo = linear_combo

    def expand(self) -> LinearCombination:
        if is_not_expanded(self):
            derivatives: defaultdict[Variable, float] = defaultdict(float)

            while self.linear_combo:
                main_factor, main_expr = self.linear_combo.pop()
                if is_expanded(main_expr):
                    for var, factor in main_expr.linear_combo.items():
                        derivatives[var] += main_factor * factor
                elif is_not_expanded(main_expr):
                    for factor, expr in main_expr.linear_combo:
                        self.linear_combo.append((main_factor * factor, expr))
                else:
                    raise NotImplementedError("Got unexpected type")
            res = dict(derivatives)
            return LinearCombination(res)
        else:
            return self

    def __getstate__(self) -> tuple[LinearComboType]:
        return (self.linear_combo,)

    def __setstate__(self, state: tuple[LinearComboType]) -> None:
        (self.linear_combo,) = state


class ExpandedLinearCombination(LinearCombination):
    linear_combo: ExpandedLinearComboType


class NonExpandedLinearCombination(LinearCombination):
    linear_combo: NonExpandedLinearComboType


def is_expanded(lc: LinearCombination) -> TypeGuard[ExpandedLinearCombination]:
    return isinstance(lc.linear_combo, dict)


def is_not_expanded(lc: LinearCombination) -> TypeGuard[NonExpandedLinearCombination]:
    return isinstance(lc.linear_combo, list)


class AffineScalarFunc:
    __slots__ = ("_nominal_value", "_linear_part")

    def __init__(self, nominal_value, linear_part):
        self._nominal_value = float(nominal_value)

        if not isinstance(linear_part, LinearCombination):
            linear_part = LinearCombination(linear_part)
        self._linear_part = linear_part

    @property
    def nominal_value(self) -> float:
        return self._nominal_value

    n = nominal_value

    @property
    def derivatives(self) -> ExpandedLinearComboType:
        if is_not_expanded(self._linear_part):
            self._linear_part = self._linear_part.expand()
        return cast(ExpandedLinearComboType, self._linear_part.linear_combo)

    def error_components(self) -> ExpandedLinearComboType:
        error_components: ExpandedLinearComboType = {}

        for variable, derivative in self.derivatives.items():
            if variable._std_dev == 0:
                error_components[variable] = 0
            else:
                error_components[variable] = abs(derivative * variable._std_dev)

        return error_components

    @property
    def std_dev(self) -> float:
        return float(
            np.sqrt(sum(delta**2 for delta in self.error_components().values()))
        )

    s = std_dev

    def __repr__(self):
        std_dev = self.std_dev
        if std_dev:
            std_dev_str = repr(std_dev)
        else:
            std_dev_str = "0"

        return f"{self.nominal_value:r}±{std_dev_str}"

    def __str__(self):
        return self.format("")

    def format(self, format_spec):
        return format_float(self, format_spec)

    def std_score(self, value: float):
        try:
            return (value - self._nominal_value) / self.std_dev
        except ZeroDivisionError:
            raise ValueError("The standard deviation is zero: undefined result")

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
    __slots__ = "_std_dev"

    def __init__(self, value: float, std_dev: float | None = None):
        value = float(value)
        super().__init__(value, LinearCombination({self: 1.0}))

        self.std_dev = std_dev if std_dev is not None else 0

    @property
    def std_dev(self):
        return self._std_dev

    @std_dev.setter
    def std_dev(self, std_dev):
        if std_dev < 0 and np.isfinite(std_dev):
            raise NegativeStdDev("The standard deviation cannot be negative")

        self._std_dev = float(std_dev)

    def __repr__(self):
        num_repr = super().__repr__()
        return num_repr

    def __hash__(self):
        return id(self)

    def __copy__(self):
        return Variable(self.nominal_value, self.std_dev)

    def __deepcopy__(self, memo):
        return self.__copy__()
