from __future__ import division  # Many analytical derivatives depend on this

import collections
import copy
import functools
from builtins import object, range, str, zip
from math import isfinite, sqrt  # Optimization: no attribute look-up
from warnings import warn

import numpy as np
from uncertainties.formatting import format_ufloat
from uncertainties.parsing import str_to_number_with_uncert


def correlated_values(nom_values, covariance_mat, tags=None):
    std_devs = np.sqrt(np.diag(covariance_mat))

    norm_vector = std_devs.copy()
    norm_vector[norm_vector == 0] = 1

    return correlated_values_norm(
        list(zip(nom_values, std_devs)),
        covariance_mat / norm_vector / norm_vector[:, np.newaxis],
        tags,
    )


def correlated_values_norm(values_with_std_dev, correlation_mat, tags=None):
    if tags is None:
        tags = (None,) * len(values_with_std_dev)

    (nominal_values, std_devs) = np.transpose(values_with_std_dev)

    (variances, transform) = np.linalg.eigh(correlation_mat)

    variances[variances < 0] = 0.0

    variables = tuple(
        # The variables represent "pure" uncertainties:
        Variable(0, sqrt(variance), tag)
        for (variance, tag) in zip(variances, tags)
    )

    # The coordinates of each new uncertainty as a function of the
    # new variables must include the variable scale (standard deviation):
    transform *= std_devs[:, np.newaxis]

    # Representation of the initial correlated values:
    values_funcs = tuple(
        AffineScalarFunc(value, LinearCombination(dict(zip(variables, coords))))
        for (coords, value) in zip(transform, nominal_values)
    )

    return values_funcs


def correlation_matrix(nums_with_unc):
    cov_mat = np.array(covariance_matrix(nums_with_unc))

    std_devs = np.sqrt(cov_mat.diagonal())

    return cov_mat / std_devs / std_devs[np.newaxis].T
class AffineScalarFunc(object):
    __slots__ = ("_nominal_value", "_linear_part")
    class dtype(object):
        type = staticmethod(lambda value: value)
    def __init__(self, nominal_value, linear_part):
        self._nominal_value = float(nominal_value)
        if not isinstance(linear_part, LinearCombination):
            linear_part = LinearCombination(linear_part)
        self._linear_part = linear_part
    @property
    def nominal_value(self):
        "Nominal value of the random number."
        return self._nominal_value
    n = nominal_value
    @property
    def derivatives(self):
        if not self._linear_part.expanded():
            self._linear_part.expand()
            self._linear_part.linear_combo.default_factory = None

        return self._linear_part.linear_combo

    def error_components(self):
        error_components = {}

        for variable, derivative in self.derivatives.items():
            if variable._std_dev == 0:
                error_components[variable] = 0
            else:
                error_components[variable] = abs(derivative * variable._std_dev)

        return error_components

    @property
    def std_dev(self):
        return float(sqrt(sum(delta**2 for delta in self.error_components().values())))
    s = std_dev

    def __repr__(self):
        std_dev = self.std_dev  # Optimization, since std_dev is calculated
        if std_dev:
            std_dev_str = repr(std_dev)
        else:
            std_dev_str = "0"

        return "%r+/-%s" % (self.nominal_value, std_dev_str)

    def __str__(self):
        return self.format("")

    @set_doc(format_ufloat.__doc__)
    def __format__(self, format_spec):
        return format_ufloat(self, format_spec)

    @set_doc("""
        Return the same result as self.__format__(format_spec), or
        equivalently as the format(self, format_spec) of Python 2.6+.

        This method is meant to be used for formatting numbers with
        uncertainties in Python < 2.6, with '... %s ...' %
        num.format('.2e').
        """)
    def format(self, format_spec):
        return format_ufloat(self, format_spec)

    def std_score(self, value):
        try:
            return (value - self._nominal_value) / self.std_dev
        except ZeroDivisionError:
            raise ValueError("The standard deviation is zero:" " undefined result")

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
                all_slots.add(slot_names)  # Single name
            else:
                all_slots.update(slot_names)
        # The slot values are stored:
        for name in all_slots:
            try:
                all_attrs[name] = getattr(self, name)
            except AttributeError:
                pass  # Undefined slot attribute

        return all_attrs

    def __setstate__(self, data_dict):
        """
        Hook for the pickle module.
        """
        for name, value in data_dict.items():
            setattr(self, name, value)


class LinearCombination(object):
    __slots__ = "linear_combo"

    def __init__(self, linear_combo):
        self.linear_combo = linear_combo

    def __bool__(self):
        return bool(self.linear_combo)

    def expanded(self):
        return isinstance(self.linear_combo, dict)

    def expand(self):
        derivatives = collections.defaultdict(float)

            (main_factor, main_expr) = self.linear_combo.pop()

            if main_expr.expanded():
                for var, factor in main_expr.linear_combo.items():
                    derivatives[var] += main_factor * factor

            else:  # Non-expanded form
                for factor, expr in main_expr.linear_combo:
                    # The main_factor is applied to expr:
                    self.linear_combo.append((main_factor * factor, expr))

            # print "DERIV", derivatives

        self.linear_combo = derivatives

    def __getstate__(self):
        # Not false, otherwise __setstate__() will not be called:
        return (self.linear_combo,)

    def __setstate__(self, state):
        (self.linear_combo,) = state
