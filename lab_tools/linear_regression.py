from numpy.typing import ArrayLike
import numpy as np
from scipy import stats
from typing import Iterator, Optional, Tuple


class LinearRegression:
    def __init__(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        limits: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Initialize the LinearRegression object.

        Args:
            X (ArrayLike): Array of x values (can be a list, tuple, or np.ndarray).
            Y (ArrayLike): Array of y values (can be a list, tuple, or np.ndarray).
            limits (Optional[Tuple[float, float]], optional): Minimum and maximum values of x for the regression line. Defaults to None.

        Raises:
            ValueError: If `X` and `Y` are not the same size.
        """
        self._X: np.ndarray = np.asarray(X)
        self._Y: np.ndarray = np.asarray(Y)

        if len(self._X) != len(self._Y):
            raise ValueError("X and Y must have the same length.")

        # Handling the optional limits parameter
        if limits is not None:
            self._X_min: float = limits[0]
            self._X_max: float = limits[1]
        else:
            self._X_min: float = float(min(self._X))
            self._X_max: float = float(max(self._X))

        self.slope: float = 0.0
        self.intercept: float = 0.0
        self.rvalue: float = 0.0
        self.stderr: float = 0.0
        self.intercept_stderr: float = 0.0

    def fit(self) -> "LinearRegression":
        """Fits the linear regression model using the provided X and Y data.

        Returns:
            LinearRegression: The instance of the class after fitting the model.
        """
        reg: _ = stats.linregress(self._X, self._Y)  # type: ignore

        # Accessing the result attributes
        self.slope: float = reg.slope
        self.intercept: float = reg.intercept
        self.rvalue: float = reg.rvalue
        self.stderr: float = reg.stderr
        self.intercept_stderr: float = reg.intercept_stderr
        self.x: np.ndarray = np.linspace(self._X_min, self._X_max)
        self.y: np.ndarray = self.slope * self.x + self.intercept
        return self

    def predict_y(self, x: ArrayLike) -> np.ndarray:
        """Predicts the Y value for a given X using the fitted linear regression model.

        Args:
            x (ArrayLike): The x value(s) for which to predict the corresponding y value(s).

        Raises:
            ValueError: If the model has not been fitted yet (slope or intercept is 0.0).

        Returns:
            np.ndarray: The predicted y value(s) for the given x.
        """
        x = np.asarray(x)  # Convert input to np.ndarray to ensure consistency
        if self.slope == 0.0 and self.intercept == 0.0:
            raise ValueError("Model has not been fitted yet.")
        return self.slope * x + self.intercept

    def predict_x(self, y: ArrayLike) -> np.ndarray:
        """Predicts the X value for a given Y using the fitted linear regression model.

        Args:
            y (ArrayLike): The y value(s) for which to predict the corresponding x value(s).

        Raises:
            ValueError: If the model has not been fitted yet (slope or intercept is 0.0).

        Returns:
            np.ndarray: The predicted x value(s) for the given y.
        """
        y = np.asarray(y)  # Convert input to np.ndarray to ensure consistency
        if self.slope == 0.0 and self.intercept == 0.0:
            raise ValueError("Model has not been fitted yet.")
        return  (y - self.intercept)/self.slope

    def __str__(self) -> str:
        """Returns a string representation of the fitted linear regression model.

        Returns:
            str: A string containing the slope, intercept, standard error of the slope, standard error of the intercept, and the correlation coefficient (r-value).
        """
        return (
            f"LinearRegression(slope={self.slope:.4f}, intercept={self.intercept:.4f}, "
            f"stderr={self.stderr:.4f}, intercept_stderr={self.intercept_stderr:.4f}, "
            f"rvalue={self.rvalue:.4f})"
        )

    def __iter__(self) -> Iterator[float]:
        """Allows iteration over key attributes of the LinearRegression instance."""
        yield self.slope
        yield self.intercept
        yield self.stderr
        yield self.intercept_stderr
        yield self.rvalue
