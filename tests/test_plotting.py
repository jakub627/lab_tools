import pytest
import numpy as np
import matplotlib.pyplot as plt
from lab_tools.plotting import (
    plot_bg,
    plot_errorbar,
    plot_plot,
    plot_scatter,
    interpolate,
)


@pytest.fixture
def sample_data():
    """Fixture with sample data for testing."""
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2.1, 4.2, 6.3, 8.4, 10.5])
    return x, y


def test_plot_bg(sample_data):
    """Test for the background plot setup function."""
    # Test basic plot background creation
    plot_bg(xlabel="X-Axis", ylabel="Y-Axis", xformat=2, yformat=2, isLegend=True)
    rgba = plt.gca().get_facecolor()
    color_val = "".join([f"{int(x*255):2X}" for x in rgba if x != rgba[-1]])  # type: ignore
    bg_color = f"#{color_val}"

    # Check that the plot was created successfully
    assert plt.gca().get_xlabel() == "X-Axis"
    assert plt.gca().get_ylabel() == "Y-Axis"
    assert bg_color == "#F4F4F4"  # Default background color


def test_interpolate(sample_data):
    """Test for the interpolation function."""
    x, y = sample_data
    x_new, y_new = interpolate(x, y)

    # Check if the interpolated x and y arrays have the correct length
    assert len(x_new) == 300  # Default npoints is 300
    assert len(y_new) == 300  # Check if the y array has the same length

    # Check if the interpolated points are within the range of original data
    assert np.min(x_new) >= np.min(x)
    assert np.max(x_new) <= np.max(x)

    # Check that y_new is interpolated correctly (for example, checking smoothness)
    assert np.all(np.diff(y_new) > 0)  # For this case, y should be increasing


def test_invalid_plot_bg_xlim_type():
    """Test invalid xlim argument for plot_bg function."""
    with pytest.raises(TypeError):
        plot_bg(xlim="invalid")  # type: ignore


def test_invalid_plot_bg_ylim_type():
    """Test invalid ylim argument for plot_bg function."""
    with pytest.raises(TypeError):
        plot_bg(ylim="invalid")  # type: ignore
