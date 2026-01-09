from collections import Counter
import math
from decimal import Decimal, ROUND_HALF_UP


class ErrNum:
    def __init__(self, value: float, uncertainty: float = 0.0, deps=None):
        self.v = float(value)
        self.u = float(uncertainty)
        # Bazowe zależności: Counter obiektów -> liczba wystąpień
        self._deps = Counter()
        if deps is None:
            self._deps[self] = 1
        else:
            self._deps.update(deps)

    # -------------------- pomocnicze --------------------
    @staticmethod
    def _to_ErrNum(other):
        if isinstance(other, ErrNum):
            return other
        return ErrNum(float(other), 0.0)

    def _propagate_unc(self, deps: Counter, func_derivatives=None):
        """
        Propagacja niepewności:
        - deps: Counter bazowych obiektów -> liczba ich wystąpień
        - func_derivatives: dict obiekt->pochodna względem tego obiektu (dla funkcji)
        """
        total = 0.0
        for obj_i, n_i in deps.items():
            deriv_i = func_derivatives[obj_i] if func_derivatives else 1.0
            for obj_j, n_j in deps.items():
                deriv_j = func_derivatives[obj_j] if func_derivatives else 1.0
                # pełna zależność dla tego samego obiektu, zero jeśli różne obiekty
                cov = obj_i.u * obj_j.u if obj_i is obj_j else 0.0
                total += n_i * n_j * deriv_i * deriv_j * cov
        return math.sqrt(total)

    # -------------------- operatory --------------------
    def __add__(self, other):
        other = self._to_ErrNum(other)
        new_value = self.v + other.v
        new_deps = self._deps + other._deps
        new_unc = self._propagate_unc(new_deps)
        return ErrNum(new_value, new_unc, deps=new_deps)

    def __sub__(self, other):
        other = self._to_ErrNum(other)
        new_value = self.v - other.v
        new_deps = self._deps + other._deps
        new_unc = self._propagate_unc(new_deps)
        return ErrNum(new_value, new_unc, deps=new_deps)

    def __mul__(self, other):
        other = self._to_ErrNum(other)
        new_value = self.v * other.v
        new_deps = self._deps + other._deps
        # względne pochodne dla mnożenia: df/dx = y, df/dy = x
        derivs = {
            **{obj: other.v for obj in self._deps},
            **{obj: self.v for obj in other._deps},
        }
        new_unc = self._propagate_unc(new_deps, func_derivatives=derivs)
        return ErrNum(new_value, new_unc, deps=new_deps)

    def __truediv__(self, other):
        other = self._to_ErrNum(other)
        new_value = self.v / other.v
        new_deps = self._deps + other._deps
        derivs = {
            **{obj: 1 / other.v for obj in self._deps},
            **{obj: -self.v / (other.v**2) for obj in other._deps},
        }
        new_unc = self._propagate_unc(new_deps, func_derivatives=derivs)
        return ErrNum(new_value, new_unc, deps=new_deps)

    def __pow__(self, other):
        other = self._to_ErrNum(other)
        new_value = self.v**other.v
        new_deps = self._deps + other._deps
        derivs = {
            **{obj: other.v * self.v ** (other.v - 1) for obj in self._deps},
            **{obj: new_value * math.log(self.v) for obj in other._deps},
        }
        new_unc = self._propagate_unc(new_deps, func_derivatives=derivs)
        return ErrNum(new_value, new_unc, deps=new_deps)

    # -------------------- operatory odwrotne dla skalara --------------------
    __radd__ = __add__
    __rsub__ = lambda self, other: ErrNum._to_ErrNum(other).__sub__(self)
    __rmul__ = __mul__
    __rtruediv__ = lambda self, other: ErrNum._to_ErrNum(other).__truediv__(self)

    # -------------------- reprezentacja --------------------
    def __repr__(self):
        return f"ErrNum(v={self.v}, u={self.u})"
