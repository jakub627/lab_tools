import pytest
import numpy as np
from lab_tools.checking import check_if_equal_extended_unc


def test_equal_within_uncertainty():
    """Test for values that are equal within the extended uncertainty bounds."""
    # Case where the two values are equal and uncertainties are the same
    x1 = 10
    x2 = 10
    u_x1 = 0.5
    u_x2 = 0.5

    # The function should print the message indicating equality within bounds
    # This test doesn't assert the print output but checks if no exceptions are raised
    try:
        check_if_equal_extended_unc(x1, x2, u_x1, u_x2)
    except Exception as e:
        pytest.fail(f"Unexpected error raised: {e}")


def test_not_equal_within_uncertainty():
    """Test for values that are not equal within the extended uncertainty bounds."""
    # Case where the two values are not equal even when uncertainties are accounted for
    x1 = 10
    x2 = 12
    u_x1 = 0.5
    u_x2 = 0.5

    # The function should print the message indicating inequality within bounds
    # This test doesn't assert the print output but checks if no exceptions are raised
    try:
        check_if_equal_extended_unc(x1, x2, u_x1, u_x2)
    except Exception as e:
        pytest.fail(f"Unexpected error raised: {e}")


def test_type_error_for_invalid_input():
    """Test for invalid input types."""
    # Case where x1 is a string, which should raise a TypeError
    with pytest.raises(TypeError):
        check_if_equal_extended_unc("10", 10, 1, 1) # type: ignore

    # Case where u_x1 is a string, which should raise a TypeError
    with pytest.raises(TypeError):
        check_if_equal_extended_unc(10, 10, "1", 1) # type: ignore

    # Case where x2 is a list, which should raise a TypeError
    with pytest.raises(TypeError):
        check_if_equal_extended_unc(10, [10], 1, 1) # type: ignore


def test_edge_case_zero_uncertainty():
    """Test for edge case where one uncertainty is zero."""
    x1 = 10
    x2 = 10
    u_x1 = 0
    u_x2 = 1

    # Should be considered equal because the uncertainty of x1 is zero
    try:
        check_if_equal_extended_unc(x1, x2, u_x1, u_x2)
    except Exception as e:
        pytest.fail(f"Unexpected error raised: {e}")


def test_large_uncertainty():
    """Test for a case with large uncertainty values."""
    x1 = 10
    x2 = 15
    u_x1 = 10
    u_x2 = 10

    # Given the large uncertainties, the difference should be within bounds
    try:
        check_if_equal_extended_unc(x1, x2, u_x1, u_x2)
    except Exception as e:
        pytest.fail(f"Unexpected error raised: {e}")
