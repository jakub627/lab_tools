from typing import Iterator
import numpy as np
from collections.abc import Callable
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
from scipy.optimize import fsolve


class FitCurve(object):
    """
    A class to perform curve fitting using scipy's curve_fit function. It calculates the best-fit parameters and their uncertainties, along with the coefficient of determination (R²).

    Attributes:
        _x (np.ndarray): The independent variable data.
        _y (np.ndarray): The dependent variable data.
        _fun (Callable): The model function to fit.
        args (np.ndarray): The optimal parameters for the model function.
        args_stderr (np.ndarray): The standard errors (uncertainties) for the parameters.
        r2 (float): The coefficient of determination (R²).
        x (np.ndarray): The range of x values for plotting the fitted curve.
        y (np.ndarray): The fitted y values for plotting.
    """

    def __init__(
        self,
        fun: Callable,
        x: ArrayLike,
        y: ArrayLike,
        limits: tuple[float, float] | None = None,
    ) -> None:
        """
        Initialize the FitCurve class with the function to fit and data.

        Args:
            fun (Callable): The model function to fit.
            x (ArrayLike): The independent variable data.
            y (ArrayLike): The dependent variable data.
            limits (tuple, optional): The min and max limits for the x values, used for plotting.
                Defaults to None, which will use the data bounds.

        Raises:
            ValueError: If the lengths of x and y do not match.
        """

        self._fun = fun
        self._x = np.atleast_1d(x)
        self._y = np.atleast_1d(y)
        self._x_min = limits[0] if limits is not None else np.min(x)
        self._x_max = limits[1] if limits is not None else np.max(x)
        self._fitted = False

        if len(self._x) != len(self._y):
            raise ValueError("X and Y must have the same length.")

    def fit(self) -> "FitCurve":
        """
        Perform the curve fitting process using scipy's curve_fit and calculate the
        optimal parameters, their uncertainties, and the coefficient of determination (R²).

        Returns:
            FitCurve: The instance of the FitCurve class with fitted parameters and results.
        """
        try:
            # Attempt to fit the curve
            self.popt, self.pcov = curve_fit(self._fun, self._x, self._y)
            self.args = self.popt

            # Calculate fitted values
            self._y_fit = self._fun(self._x, *self.popt)

            # Calculate R²
            self._ss_res = np.sum((self._y - self._y_fit) ** 2)
            self._ss_tot = np.sum((self._y - np.mean(self._y)) ** 2)
            self.r2 = 1 - (self._ss_res / self._ss_tot)

            # Calculate standard errors (uncertainties)
            self.args_stderr = np.sqrt(np.diag(self.pcov))

            # Check for inf in standard errors (could not estimate covariance)
            if np.any(np.isnan(self.args_stderr)) or np.any(np.isinf(self.args_stderr)):
                warnings.warn(
                    "Covariance of parameters could not be estimated. Uncertainties set to 'inf'.",
                    OptimizeWarning,
                )
                # Optionally, set a fallback value for stderr (for example, a large value or None)
                self.args_stderr = np.full_like(self.args_stderr, np.inf)

        except (RuntimeError, OptimizeWarning) as e:
            warnings.warn(f"Curve fitting failed: {e}")
            self.args = np.full_like(
                self._x, np.nan
            )  # Set parameters to NaN if fitting fails
            self.args_stderr = np.full_like(self._x, np.nan)  # Uncertainty set to NaN
            self.r2 = np.nan
            self._fitted = False
            return self

        # Generate fitted curve
        self.x = np.linspace(self._x_min, self._x_max, 100)
        self.y = self._fun(self.x, *self.args)

        self._fitted = True
        return self

    def __str__(self) -> str:
        """
        Return a string representation of the fitted curve with parameters, uncertainties, and R².

        Returns:
            str: A formatted string showing the fitted parameters, their uncertainties, and R².
        """
        formatted_args = ", ".join(f"{arg:.4f}" for arg in self.args)
        formatted_stderr = ", ".join(f"{stderr:.4f}" for stderr in self.args_stderr)
        return (
            f"FitCurve(args=[{formatted_args}], "
            f"args_stderr=[{formatted_stderr}], r2={self.r2:.4f})"
        )

    def __iter__(self) -> Iterator[float]:
        """
        Iterate over the fitted parameters, their uncertainties, and R².

        Yields:
            float: Each element from the parameters, their uncertainties, and the R² value.
        """
        for arg in self.args:
            yield arg
        for err in self.args_stderr:
            yield err
        yield self.r2

    def predict_y(self, x: ArrayLike | float | int) -> np.ndarray:
        """
        Predict the y values based on the given x values using the fitted model.

        Args:
            x (Union[ArrayLike, float, int]): The independent variable values (either a single value or an array)
                for which to predict the corresponding y values.

        Returns:
            np.ndarray: The predicted y values based on the model parameters.
        """
        x = np.asarray(x)
        return self._fun(x, *self.args)

    def predict_x(self, y: ArrayLike | float | int) -> np.ndarray:
        """
        Predict the x values based on the given y values using the inverse of the fitted model.
        This method uses numerical solving to find x values that satisfy the equation y = f(x).

        Args:
            y (ArrayLike): The dependent variable values for which to predict the corresponding x values.

        Returns:
            np.ndarray: The predicted x values based on the model parameters.
        """

        def model_inverse(x, y_target):
            return self._fun(x, *self.args) - y_target

        y = np.atleast_1d(y)

        x_predicted = []
        for yi in y:
            initial_guess = np.mean(self._x)

            root = fsolve(model_inverse, x0=initial_guess, args=(yi,))

            if np.isnan(root[0]) or np.isinf(root[0]):
                x_predicted.append(np.nan)
            else:
                x_predicted.append(root[0])

        return np.array(x_predicted)
