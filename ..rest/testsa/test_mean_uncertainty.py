import pytest
import numpy as np
from lab_tools.mean_uncertainty import MeanUncertainty


def test_mean_uncertainty_basic():
    """Test for calculating mean and standard error of the mean (stderr)."""
    # Sample data
    data = [1.5, 2.5, 3.0, 4.0, 5.0]

    # Create an instance of MeanUncertainty
    mu = MeanUncertainty(data)

    # Check if the mean and stderr are calculated correctly
    assert mu.mean == pytest.approx(3.2, rel=1e-2), f"Expected mean 3.2, got {mu.mean}"
    assert mu.stderr == pytest.approx(
        0.6041, rel=1e-2
    ), f"Expected stderr ~0.6041, got {mu.stderr}"


def test_mean_uncertainty_empty_data():
    """Test for empty data, should raise an exception."""
    data = []

    # Should raise ValueError since the input data is empty
    with pytest.raises(
        ValueError
    ):
        MeanUncertainty(data)


def test_mean_uncertainty_single_value():
    """Test for a single data point."""
    data = [5.0]

    # Check if mean is correct and stderr is zero (since there is no variance in a single value)
    with pytest.raises(ValueError):
        MeanUncertainty(data)


def test_mean_uncertainty_non_numeric_data():
    """Test for non-numeric data, should raise an exception."""
    data = ["a", "b", "c"]

    # Should raise TypeError since the input data is non-numeric
    with pytest.raises(
        TypeError,
        
    ):
        MeanUncertainty(data)


def test_mean_uncertainty_string_representation():
    """Test for string representation of the MeanUncertainty object."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]

    # Create an instance of MeanUncertainty
    mu = MeanUncertainty(data)

    # Check if the string representation is correctly formatted
    str_repr = str(mu)
    assert "mean=" in str_repr, "String representation should include 'mean'."
    assert "stderr=" in str_repr, "String representation should include 'stderr'."
