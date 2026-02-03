from typing import Iterator, Self
from numpy.typing import ArrayLike
import numpy as np
from numpy import float64, ndarray, dtype
import pandas as pd

from .exceptions import NotFittedError
from .validate import Validate


class LinearRegression:

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        limits: ArrayLike | None = None,
        force_zero: bool = False,
    ) -> None:
        """
        Initialize the LinearRegression object

        Parameters
        ----------
        x : ArrayLike
            Array of x values
        y : ArrayLike
            Array of y values
        limits : ArrayLike | None, optional
            Minimum and maximum values of x for the regression line, by default None
        force_zero : bool, optional
            Wether to force intercept to be zero, by default False

        Raises
        ------
        ValueError
            If inputted arrays are not 1D
        ValueError
            If inputted arrays are not the same size
        ValueError
            If limits is wrong type
        """
        self.x = np.asarray(x, dtype=float64)
        self.y = np.asarray(y, dtype=float64)
        self.force_zero = force_zero
        self.fitted = False

        Validate.is_1d_array(self.x)
        Validate.is_1d_array(self.y)
        Validate.arrays_same_length(self.x, self.y)

        if limits is not None:
            Validate.limits_type(limits)
            self.limits = tuple(np.asarray(limits))
        else:
            self.limits = (self.x.min(), self.x.max())

        self.slope: float = 0.0
        self.intercept: float = 0.0
        self.rvalue: float = 0.0
        self.stderr: float = 0.0
        self.intercept_stderr: float = 0.0

    def fit(self) -> Self:
        """
        Fits the linear regression model using the provided X and Y data
        """
        self.slope: float = 0.0
        self.intercept: float = 0.0
        self.stderr: float = 0.0
        self.intercept_stderr: float = 0.0

        n = len(self.x)
        if self.force_zero:
            S_xx = np.sum(self.x**2)
            self.slope = np.sum(self.x * self.y) / S_xx
            self.intercept = 0.0
            y_fit = self.slope * self.x
            sigma2 = np.sum((self.y - y_fit) ** 2) / (n - 1)
            self.stderr = np.sqrt(sigma2 / S_xx)
            self.intercept_stderr = 0.0
        else:
            x_mean = np.mean(self.x, dtype=float64)
            y_mean = np.mean(self.y, dtype=float64)
            S_xx = np.sum((self.x - x_mean) ** 2)
            self.slope = np.sum((self.x - x_mean) * (self.y - y_mean)) / S_xx
            self.intercept = y_mean - self.slope * x_mean
            y_fit = self.slope * self.x + self.intercept
            sigma2 = np.sum((self.y - y_fit) ** 2) / (n - 2)
            self.stderr = np.sqrt(sigma2 / S_xx)
            self.intercept_stderr = np.sqrt(sigma2 * (1 / n + x_mean**2 / S_xx))
        self.rvalue = np.corrcoef(self.x, self.y)[0, 1]

        self.x_fit = np.linspace(self.limits[0], self.limits[1], dtype=float64)
        self.y_fit = self.slope * self.x + self.intercept
        self.fitted = True
        return self

    def predict_y(self, x: ArrayLike) -> ndarray[tuple[int], dtype[float64]]:
        """
        Predicts the y value for a given x using the fitted linear regression model

        Parameters
        ----------
        x : ArrayLike
            The x value(s) for which to predict the corresponding y value(s)

        Returns
        -------
        ndarray[tuple[int], dtype[float64]]
            The predicted y value(s) for the given x

        Raises
        ------
        ValueError
            If model has ambiguous parameters
        """
        if not self.fitted:
            raise NotFittedError()
        x = np.asarray(x)
        if self.slope == 0.0 and self.intercept == 0.0:
            raise ValueError("Model parameters are ambiguous")

        y = self.slope * x + self.intercept
        return y

    def predict_x(self, y: ArrayLike) -> ndarray[tuple[int], dtype[float64]]:
        """
        Predicts the x value for a given y using the fitted linear regression model

        Parameters
        ----------
        y : ArrayLike
            The y value(s) for which to predict the corresponding x value(s)

        Returns
        -------
        ndarray[tuple[int], dtype[float64]]
            The predicted x value(s) for the given y

        Raises
        ------
        ValueError
            If model has ambiguous parameters
        """
        if not self.fitted:
            raise NotFittedError
        y = np.asarray(y)
        if self.slope == 0.0 and self.intercept == 0.0:
            raise ValueError("Model parameters are ambiguous")

        x = (y - self.intercept) / self.slope
        return x

    def __str__(self) -> str:
        if self.fitted:
            txt = ", ".join(
                [
                    f"slope={self.slope:g}",
                    f"intercept={self.intercept:g}",
                    f"stderr={self.stderr:g}",
                    f"intercept_stderr={self.intercept_stderr:g}",
                    f"rvalue={self.rvalue:g}",
                ],
            )
        else:
            txt = "not fitted"

        return f"{self.__class__.__name__}({txt})"

    __repr__ = __str__

    def __iter__(self) -> Iterator[float]:
        yield self.slope
        yield self.intercept
        yield self.stderr
        yield self.intercept_stderr
        yield self.rvalue

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the linear regression parameters to a pandas DataFrame

        Returns
        -------
        DataFrame
            DataFrame containing the linear regression parameters
        """
        if not self.fitted:
            raise NotFittedError
        a = float(self.slope)
        ua = float(self.stderr)
        b = float(self.intercept)
        ub = float(self.intercept_stderr)
        rval = float(self.rvalue)
        r2 = float(self.rvalue**2)

        return pd.DataFrame(
            {
                "slope": [a, ua],
                "intercept": [b, ub],
                "r_squared": [r2, 0.0],
                "rvalue": [rval, 0.0],
            },
            index=["x", "u_x"],
            dtype=np.float64,
        )


def main():
    reg = LinearRegression([1], [1])
    print(reg)


if __name__ == "__main__":
    main()
