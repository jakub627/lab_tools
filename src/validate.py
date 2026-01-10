import numpy as np
from numpy.typing import ArrayLike

from dtypes import FileDescriptorOrPath


class Validate:
    @staticmethod
    def limits_type(limits: ArrayLike) -> None:
        arr = np.asarray(limits)

        if not np.issubdtype(arr.dtype, np.number) or arr.size != 2:
            raise ValueError("Limits must be a sequence of two numeric values")

    @staticmethod
    def is_1d_array(arr: ArrayLike):
        arr = np.asarray(arr)
        if arr.ndim != 1:
            raise ValueError("Input arrays must be one-dimensional")

    @staticmethod
    def arrays_same_length(arr1: ArrayLike, arr2: ArrayLike):
        arr1 = np.asarray(arr1)
        arr2 = np.asarray(arr2)
        if len(arr1) != len(arr2):
            raise ValueError("Input arrays must have the same length")

    @staticmethod
    def file_extension(file: FileDescriptorOrPath, expected_ext: str) -> None:
        if isinstance(file, int):
            return

        if not expected_ext.startswith("."):
            raise ValueError("expected_ext must start with '.' (e.g. '.csv')")

        file_str = str(file).lower()
        expected_ext = expected_ext.lower()

        if not file_str.endswith(expected_ext):
            ext = (
                "." + file_str.rsplit(".", 1)[-1]
                if "." in file_str
                else "<no extension>"
            )
            raise ValueError(
                f"Invalid file type: expected '{expected_ext}', got '{ext}'"
            )


def main():
    v = np.array([[2], [3]])
    print(v[0])


if __name__ == "__main__":
    main()
