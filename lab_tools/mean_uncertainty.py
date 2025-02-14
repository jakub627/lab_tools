from typing import Any, Iterator
import numpy as np
from numpy.typing import ArrayLike


class MeanUncertainty:
    """Calculates the average of the given data and computes the standard uncertainty of the mean.

    This class takes a numerical array-like object (such as a list, tuple, or NumPy array),
    calculates the mean, and computes the standard error of the mean (SEM).
    The SEM is defined as the standard deviation of the sample divided by the square root of the sample size.

    :param data: A numerical array-like object (int or float).
    :type data: ArrayLike
    :return: An instance of MeanUncertainty containing the mean and stderr
    :rtype: MeanUncertainty
    """

    def __init__(self, data: ArrayLike) -> None:
        data = np.asarray(data)
        if len(data) <= 1:
            raise ValueError("data must have at least two elements")
        if not any([isinstance(item, (float, int)) for item in data]):
            raise TypeError("data elements must be numbers")
        n: int = len(data)
        self.mean: float = np.mean(data, axis=0)
        self.stderr: float = np.std(data, ddof=1) / np.sqrt(n)
        self.rel_err = self.stderr/self.mean

    def __str__(self) -> str:
        """Returns a string representation of the MeanUncertainty object."""
        return f"MeanUncertainty(mean={self.mean:.4f}, stderr={self.stderr:.4f}, rel_err={self.rel_err:.4f})"

    def __repr__(self) -> str:
        return self.__str__()

    def __iter__(self) -> Iterator[float]:
        """Allows iteration over the MeanUncertainty instance, yielding mean and stderr."""
        yield self.mean
        yield self.stderr
        yield self.rel_err
