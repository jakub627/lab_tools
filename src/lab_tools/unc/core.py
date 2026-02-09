from __future__ import annotations
from collections import defaultdict
import copy
from typing import Any, Callable, Sequence, TypeGuard, cast, Iterable

import numpy as np

from . import ops

from .formatting import format_float


__all__ = [
    "ufloat",
    "ufloat_fromstr",
    "nominal_value",
    "std_dev",
    "covariance_matrix",
    "UFloat",
    "Variable",
    "wrap",
    "nan_if_exception",
    "modified_operators",
    "modified_ops_with_reflection",
    "correlated_values",
    "correlated_values_norm",
    "correlation_matrix",
]


def correlated_values(
    nominal_values: Sequence[float],
    covariance_matrix: np.ndarray,
) -> tuple[AffineScalarFunc, ...]:

    nominal_values_np = np.asarray(nominal_values, dtype=float)
    covariance_matrix = np.asarray(covariance_matrix, dtype=float)

    if covariance_matrix.shape != (len(nominal_values_np),) * 2:
        raise ValueError("Covariance matrix must be square and match input size.")

    std_devs = np.sqrt(np.diag(covariance_matrix))

    norm = std_devs.copy()
    norm[norm == 0] = 1.0

    correlation_matrix = covariance_matrix / norm / norm[:, np.newaxis]

    values_with_std = list(zip(nominal_values_np, std_devs))

    return correlated_values_norm(values_with_std, correlation_matrix)


def correlated_values_norm(
    values_with_std: Sequence[tuple[float, float]],
    correlation_matrix: np.ndarray,
) -> tuple[AffineScalarFunc, ...]:

    correlation_matrix = np.asarray(correlation_matrix, dtype=float)

    n = len(values_with_std)

    if correlation_matrix.shape != (n, n):
        raise ValueError("Correlation matrix must be square and match input size.")

    nominal_values, std_devs = np.transpose(values_with_std)

    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)

    eigenvalues = np.clip(eigenvalues, 0.0, None)

    variables = tuple(Variable(0.0, np.sqrt(var)) for var in eigenvalues)

    transform = eigenvectors * std_devs[:, np.newaxis]

    result: list[AffineScalarFunc] = []

    for value, coeffs in zip(nominal_values, transform):
        lc = LinearCombination(dict(zip(variables, coeffs)))
        result.append(AffineScalarFunc(value, lc))

    return tuple(result)


def correlation_matrix(
    values: Sequence[AffineScalarFunc],
) -> np.ndarray:
    """
    Return the correlation matrix of given uncertain numbers.
    """

    cov = np.asarray(covariance_matrix(values), dtype=float)

    std = np.sqrt(np.diag(cov))
    std[std == 0] = 1.0  # avoid division by zero

    return cov / std / std[:, np.newaxis]


def nominal_value(x) -> float:
    if isinstance(x, AffineScalarFunc):
        return x.nominal_value
    else:
        return x


def std_dev(x) -> float:
    if isinstance(x, AffineScalarFunc):
        return x.std_dev
    else:
        return 0.0


def ufloat(nominal_value: float, std_dev: float | None = None) -> Variable:
    return Variable(nominal_value, std_dev)


def covariance_matrix(nums_with_unc: Sequence[AffineScalarFunc]):
    all_vars = list({var for expr in nums_with_unc for var in expr.derivatives})
    var_index = {v: i for i, v in enumerate(all_vars)}
    variances = np.array([v._std_dev**2 for v in all_vars])

    J = np.zeros((len(nums_with_unc), len(all_vars)))

    for i, expr in enumerate(nums_with_unc):
        for var, deriv in expr.derivatives.items():
            J[i, var_index[var]] = deriv

    covariance = J @ (variances * J.T)

    return covariance


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
    _to_affine_scalar: Callable[[Any], AffineScalarFunc]

    def __sub__(self, other: AffineScalarFunc | float | int) -> AffineScalarFunc: ...
    def __add__(self, other: AffineScalarFunc | float | int) -> AffineScalarFunc: ...
    def __mul__(self, other: AffineScalarFunc | float | int) -> AffineScalarFunc: ...
    def __truediv__(
        self, other: AffineScalarFunc | float | int
    ) -> AffineScalarFunc: ...
    def __mod__(self, other: AffineScalarFunc | float | int) -> AffineScalarFunc: ...
    def __div__(self, other: AffineScalarFunc | float | int) -> AffineScalarFunc: ...
    def __pow__(self, other: AffineScalarFunc | float | int) -> AffineScalarFunc: ...
    def __rpow__(self, other: AffineScalarFunc | float | int) -> AffineScalarFunc: ...

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
        return self.__format__("")

    def __format__(self, format_spec: str) -> str:
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


ops.add_arithmetic_ops(AffineScalarFunc)
ops.add_comparative_ops(AffineScalarFunc)

to_affine_scalar: Callable[[Any], AffineScalarFunc] = AffineScalarFunc._to_affine_scalar
