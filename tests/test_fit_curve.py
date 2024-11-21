import pytest
import numpy as np
from lab_tools.fit_curve import FitCurve


# Define a simple linear model function for testing
def linear_model(x, a, b):
    return a * x + b


# Test Initialization and Validation
def test_fitcurve_initialization():
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 3, 4, 5, 6])

    # Valid initialization
    curve = FitCurve(fun=linear_model, x=x_data, y=y_data)
    assert isinstance(curve, FitCurve)
    assert np.array_equal(curve._x, x_data)
    assert np.array_equal(curve._y, y_data)

    # Invalid initialization: x and y have different lengths
    with pytest.raises(ValueError):
        FitCurve(fun=linear_model, x=x_data, y=np.array([1, 2, 3]))


# Test Curve Fitting
def test_curve_fitting():
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 6, 8, 10])  # Linear data with a=2 and b=0

    curve = FitCurve(fun=linear_model, x=x_data, y=y_data)
    curve.fit()

    assert np.isclose(curve.args[0], 2, atol=0.1)  # a ≈ 2
    assert np.isclose(curve.args[1], 0, atol=0.1)  # b ≈ 0
    assert np.isclose(curve.r2, 1.0, atol=0.01)  # R² ≈ 1 (perfect fit)


# Test R² Calculation
def test_r2_calculation():
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 6, 8, 10])  # Linear data

    curve = FitCurve(fun=linear_model, x=x_data, y=y_data)
    curve.fit()

    assert np.isclose(curve.r2, 1.0, atol=0.01)  # Perfect linear fit


# Test String Representation
def test_fitcurve_str():
    x_data = np.array([0, 1, 2, 3, 4, 5, 6])
    y_data = np.array([0, 2, 4, 6, 8, 10, 12])  # Linear data

    curve = FitCurve(fun=linear_model, x=x_data, y=y_data)
    curve.fit()

    # Test that the string representation includes the parameters and R²
    assert "args=[2.0000, 0.0000], " in str(curve)
    assert "r2=1.0000" in str(curve)


# Test Predictions
def test_predict_y():
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 6, 8, 10])  # Linear data

    curve = FitCurve(fun=linear_model, x=x_data, y=y_data)
    curve.fit()

    # Test prediction for a single value
    y_pred = curve.predict_y(6)
    assert np.isclose(y_pred, 12)  # Should return 12 for x=6

    # Test prediction for an array
    y_pred_array = curve.predict_y([6, 7])
    assert np.allclose(y_pred_array, [12, 14])  # Should return [12, 14]


def test_predict_x():
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 6, 8, 10])  # Linear data

    curve = FitCurve(fun=linear_model, x=x_data, y=y_data)
    curve.fit()

    # Test prediction for a single y value
    x_pred = curve.predict_x(12)
    assert np.isclose(x_pred, 6)  # Should return 6 for y=12

    # Test prediction for an array of y values
    x_pred_array = curve.predict_x([12, 14])
    assert np.allclose(x_pred_array, [6, 7])  # Should return [6, 7]


# Test Iteration
def test_fitcurve_iteration():
    x_data = np.array([0, 1, 2, 3, 4, 5])
    y_data = np.array([0, 2, 4, 6, 8, 10])  # Linear data

    curve = FitCurve(fun=linear_model, x=x_data, y=y_data)
    curve.fit()

    # Test iteration over parameters, uncertainties, and R²
    results = [
        item
        for sublist in curve
        for item in (sublist if isinstance(sublist, list) else [sublist])
    ]

    assert len(results) == 5  # 2 parameters + 2 uncertainties + 1 R²
    assert np.isclose(results[0], 2, atol=0.1)  # First parameter (a)
    assert np.isclose(results[-1], 1, atol=0.1)  # R²
