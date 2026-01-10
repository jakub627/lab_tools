import pytest
import numpy as np
from validate import Validate


class TestValidate:
    def test_limits_type_valid(self):
        # Poprawne limits: tuple dwóch liczb
        Validate.limits_type((1.0, 2.0))
        Validate.limits_type([1, 2])
        Validate.limits_type(np.array([1.5, 2.5]))

    def test_limits_type_invalid_not_sequence(self):
        with pytest.raises(
            ValueError, match="Limits must be a sequence of two numeric values"
        ):
            Validate.limits_type(1.0)

    def test_limits_type_invalid_wrong_length(self):
        with pytest.raises(
            ValueError, match="Limits must be a sequence of two numeric values"
        ):
            Validate.limits_type((1.0,))
        with pytest.raises(
            ValueError, match="Limits must be a sequence of two numeric values"
        ):
            Validate.limits_type((1.0, 2.0, 3.0))

    def test_limits_type_invalid_not_numeric(self):
        with pytest.raises(
            ValueError, match="Limits must be a sequence of two numeric values"
        ):
            Validate.limits_type(("a", "b"))
        with pytest.raises(
            ValueError, match="Limits must be a sequence of two numeric values"
        ):
            Validate.limits_type((1.0, "b"))

    def test_is_1d_array_valid(self):
        Validate.is_1d_array([1, 2, 3])
        Validate.is_1d_array(np.array([1.0, 2.0]))
        Validate.is_1d_array((1, 2))

    def test_is_1d_array_invalid_multidimensional(self):
        with pytest.raises(ValueError, match="Input arrays must be one-dimensional"):
            Validate.is_1d_array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="Input arrays must be one-dimensional"):
            Validate.is_1d_array(np.array([[1, 2], [3, 4]]))

    def test_arrays_same_length_valid(self):
        Validate.arrays_same_length([1, 2, 3], [4, 5, 6])
        Validate.arrays_same_length(np.array([1, 2]), np.array([3, 4]))

    def test_arrays_same_length_invalid(self):
        with pytest.raises(ValueError, match="Input arrays must have the same length"):
            Validate.arrays_same_length([1, 2], [3, 4, 5])
        with pytest.raises(ValueError, match="Input arrays must have the same length"):
            Validate.arrays_same_length(np.array([1]), np.array([2, 3]))

    def test_file_extension_valid(self):
        Validate.file_extension("file.csv", ".csv")
        Validate.file_extension("FILE.TXT", ".txt")  # Wielkość liter ignorowana
        Validate.file_extension("/path/to/file.json", ".JSON")

    def test_file_extension_invalid_no_dot(self):
        with pytest.raises(ValueError, match="expected_ext must start with '.'"):
            Validate.file_extension("file.csv", "csv")

    def test_file_extension_invalid_wrong_extension(self):
        with pytest.raises(
            ValueError, match="Invalid file type: expected '.csv', got '.txt'"
        ):
            Validate.file_extension("file.txt", ".csv")

    def test_file_extension_no_extension(self):
        with pytest.raises(
            ValueError, match="Invalid file type: expected '.csv', got '<no extension>'"
        ):
            Validate.file_extension("file", ".csv")

    def test_file_extension_file_descriptor(self):
        # Dla deskryptora pliku (int), metoda nie robi nic
        Validate.file_extension(0, ".csv")  # Nie powinno rzucić wyjątku
