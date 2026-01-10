from collections.abc import Sequence
import numpy as np
from numpy.typing import ArrayLike


class Validate:
    @staticmethod
    def limits_type(limits: ArrayLike) -> None:
        if not (
            isinstance(limits, Sequence)
            and len(limits) == 2
            and all(isinstance(val, (int, float)) for val in limits)
        ):
            raise ValueError("Limits must be a sequence of two numeric values.")

    @staticmethod
    def is_1d_array(arr: ArrayLike):
        arr = np.asarray(arr)
        if arr.ndim != 1:
            raise ValueError("Input arrays must be one-dimensional.")

    @staticmethod
    def arrays_same_length(arr1: ArrayLike, arr2: ArrayLike):
        arr1 = np.asarray(arr1)
        arr2 = np.asarray(arr2)
        if len(arr1) != len(arr2):
            raise ValueError("Input arrays must have the same length.")


def main():
    Validate.limits_type(("a", "b"))


if __name__ == "__main__":
    main()
