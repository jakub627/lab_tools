from collections.abc import Iterator
from typing import Callable, Self
import warnings
import numpy as np
from numpy.typing import ArrayLike
from numpy import dtype, float64, ndarray
from scipy.optimize import curve_fit, OptimizeWarning
from exceptions import NotFittedError
from validation import Validate
from scipy.optimize import fsolve


class CurveFit:

    def __init__(
        self,
        fun: Callable,
        x: ArrayLike,
        y: ArrayLike,
        *,
        limits: ArrayLike | None = None,
        p0: ArrayLike | None = None,
        bounds: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> None:
        """
        Initialize the FitCurve class with the function to fit and data

        Parameters
        ----------
        fun : Callable
            The model function to fit
        x : ArrayLike
            The x data points
        y : ArrayLike
            The y data points
        limits : ArrayLike | None, optional
            The limits of the fit, by default None
        p0 : ArrayLike | None, optional
            The initial guess for the parameters, by default None
        bounds : tuple[ArrayLike, ArrayLike] | None, optional
            The bounds for the parameters, by default None
        """
        self.fun = fun
        self.x = np.asarray(x, dtype=float64)
        self.y = np.asarray(y, dtype=float64)
        self.p0 = p0
        self.bounds = bounds
        self.fitted = False

        Validate.is_1d_array(self.x)
        Validate.is_1d_array(self.y)
        Validate.arrays_same_length(self.x, self.y)
        if limits is not None:
            Validate.limits_type(limits)
            self.limits = tuple(np.asarray(limits))
        else:
            self.limits = (self.x.min(), self.x.max())

        self._params: np.ndarray[tuple[int], dtype[float64]] | None = None
        self._covariance: np.ndarray[tuple[int, int], dtype[float64]] | None = None
        self._r2: float | None = None

    @property
    def params(self) -> ndarray[tuple[int], dtype[float64]]:
        if not self.fitted or self._params is None:
            raise NotFittedError()
        return self._params

    @property
    def stderr(self) -> ndarray[tuple[int], dtype[float64]]:
        if not self.fitted or self._stderr is None:
            raise NotFittedError()
        return self._stderr

    @property
    def r2(self) -> float:
        if not self.fitted or self._r2 is None:
            raise NotFittedError("Model is not fitted yet.")
        return self._r2

    @property
    def covariance(self) -> ndarray[tuple[int, int], dtype[float64]]:
        if not self.fitted or self._covariance is None:
            raise NotFittedError("Model is not fitted yet.")
        return self._covariance

    def fit(self) -> Self:
        """
        Perform the curve fitting process using scipy's curve_fit and calculate the optimal parameters, their uncertainties, and the coefficient of determination (R²)

        Returns
        -------
        FitCurve
            The fitted curve object

        Raises
        ------
        NotFittedError
            If the fitting process fails
        """
        try:
            self._params, self._covariance = curve_fit(
                self.fun,
                self.x,
                self.y,
                p0=self.p0,
                bounds=self.bounds if self.bounds is not None else (-np.inf, np.inf),
            )
            if self._params is None or self._covariance is None:
                raise NotFittedError()
            self._stderr = np.sqrt(np.diag(self._covariance))
            self.x_fit = np.linspace(self.limits[0], self.limits[1], 100)
            self.y_fit = self.fun(self.x, *self._params)
            ss_res = np.sum((self.y - self.y_fit) ** 2)
            ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
            self._r2 = 1 - (ss_res / ss_tot if ss_tot != 0 else 0)
            self.fitted = True
        except (RuntimeError, OptimizeWarning) as e:
            warnings.warn(f"Curve fitting failed: {e}")
            n_params = self.fun.__code__.co_argcount - 1
            self._params = np.full(n_params, np.nan)
            self._covariance = np.full((n_params, n_params), np.nan)
            self._stderr = np.full(n_params, np.nan)
            self._r2 = np.nan
            self.fitted = False

        return self

    def predict_x(self, y: ArrayLike) -> ndarray[tuple[int, ...], dtype[float64]]:
        """
        Predict the x values based on the given y values using the inverse of the fitted model.
        This method uses numerical solving to find x values that satisfy the equation y = f(x)

        Parameters
        ----------
        y : ArrayLike
            The y values to predict x for

        Returns
        -------
        ndarray[tuple[int, ...], dtype[float64]]
            The predicted x values

        Raises
        ------
        NotFittedError
            If the curve has not been fitted yet
        """
        if not self.fitted or self._params is None:
            raise NotFittedError()
        y = np.atleast_1d(y).astype(float64)

        def model_inverse(x: float, y_target: float) -> float:
            assert self._params is not None
            return self.fun(x, *self._params) - y_target

        x0 = float(np.mean(self.x))
        x_pred = np.empty_like(y, dtype=float)
        for i in range(y.size):
            root = fsolve(model_inverse, x0=x0, args=(y[i],))
            val = root[0]
            x_pred[i] = val if np.isfinite(val) else np.nan

        return x_pred

    def predict_y(self, x: ArrayLike) -> ndarray[tuple[int, ...], dtype[float64]]:
        """
        Predict the y values based on the given x values using the fitted model

        Parameters
        ----------
        x : ArrayLike
            The x values to predict y for

        Returns
        -------
        ndarray[tuple[int, ...], dtype[float64]]
            The predicted y values

        Raises
        ------
        NotFittedError
            If the curve has not been fitted yet
        """
        if not self.fitted or self._params is None:
            raise NotFittedError()
        x = np.asarray(x, dtype=float64)
        return np.asarray(self.fun(x, *self._params), dtype=float64)

    def __iter__(self) -> Iterator[float]:
        if (
            not self.fitted
            or self._params is None
            or self._stderr is None
            or self._r2 is None
        ):
            raise NotFittedError()
        for arg in self._params:
            yield arg
        for err in self._stderr:
            yield err
        yield self._r2

    def __str__(self) -> str:
        if self.fitted and self._params is not None:
            txt = ", ".join(
                [
                    f"params=[{", ".join(f'{param:g}' for param in self._params)}]",
                    f"stderr=[{", ".join(f'{err:g}' for err in self._stderr)}]",
                    f"r2={self._r2:g}",
                ],
            )
        else:
            txt = "not fitted"

        return f"{self.__class__.__name__}({txt})"

    __repr__ = __str__


def main():
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    CurveFit(lambda x, a, b: a * x + b, x, y, limits=("a", "b"))


if __name__ == "__main__":
    main()
