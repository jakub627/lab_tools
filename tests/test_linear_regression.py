import pytest
import numpy as np
from lab_tools.linear_regression import LinearRegression


def test_linear_regression_fit():
    """Test for fitting a linear regression model."""
    # Sample data
    X = [1, 2, 3, 4, 5]
    Y = [2, 4, 5, 4, 5]

    # Create and fit the model
    model = LinearRegression(X, Y)
    model.fit()

    # Check if the model's slope and intercept are correctly calculated
    assert model.slope != 0.0, "Slope should not be zero."
    assert model.intercept != 0.0, "Intercept should not be zero."
    assert model.rvalue >= 0.0, "R-value should be non-negative."
    assert model.stderr >= 0.0, "Standard error should be non-negative."
    assert (
        model.intercept_stderr >= 0.0
    ), "Intercept standard error should be non-negative."


def test_linear_regression_predict():
    """Test for predicting Y values with the fitted model."""
    # Sample data
    X = [1, 2, 3, 4, 5]
    Y = [2, 4, 5, 4, 5]

    # Create and fit the model
    model = LinearRegression(X, Y)
    model.fit()

    # Predict for a new X value
    X_new = [6]
    Y_pred = model.predict(X_new)

    # Check if prediction is a float and matches the expected value
    assert isinstance(Y_pred, np.ndarray), "Prediction should be a numpy array."
    assert Y_pred[0] == pytest.approx(
        5.8, rel=1e-2
    ), f"Predicted value for x=6 is incorrect, got {Y_pred[0]}."


def test_linear_regression_unfitted_predict():
    """Test for predicting Y values before fitting the model (should raise an error)."""
    X = [1, 2, 3, 4, 5]
    Y = [2, 4, 5, 4, 5]

    model = LinearRegression(X, Y)

    # Predict without fitting the model
    with pytest.raises(ValueError, match="Model has not been fitted yet."):
        model.predict([6])


def test_linear_regression_string_representation():
    """Test for the string representation of the LinearRegression model."""
    X = [1, 2, 3, 4, 5]
    Y = [2, 4, 5, 4, 5]

    model = LinearRegression(X, Y)
    model.fit()

    # Check if the string representation is formatted correctly
    str_repr = str(model)
    assert "slope=" in str_repr, "String representation should include 'slope'."
    assert "intercept=" in str_repr, "String representation should include 'intercept'."
    assert "stderr=" in str_repr, "String representation should include 'stderr'."
    assert "rvalue=" in str_repr, "String representation should include 'rvalue'."


def test_linear_regression_invalid_input():
    """Test for invalid input sizes."""
    X = [1, 2, 3]
    Y = [2, 4]

    # Should raise ValueError since X and Y have different lengths
    with pytest.raises(ValueError, match="X and Y must have the same length."):
        LinearRegression(X, Y)
