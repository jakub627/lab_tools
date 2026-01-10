import numpy as np
import pandas as pd
import pytest

from exceptions import NotFittedError
from linear_regression import LinearRegression


@pytest.fixture
def simple_linear_data():
    np.random.seed(0)
    x = np.linspace(-10, 10, 50)
    y = 3.0 * x + 2.0 + np.random.normal(scale=0.5, size=len(x))
    return x, y


def test_init_accepts_arraylike(simple_linear_data):
    x, y = simple_linear_data
    reg = LinearRegression(list(x), list(y))
    assert isinstance(reg.x, np.ndarray)
    assert isinstance(reg.y, np.ndarray)


def test_init_rejects_non_1d():
    x = np.array([[1, 2, 3]])
    y = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        LinearRegression(x, y)


def test_init_rejects_different_lengths():
    x = np.array([1, 2, 3])
    y = np.array([1, 2])
    with pytest.raises(ValueError):
        LinearRegression(x, y)


def test_init_limits_validation(simple_linear_data):
    x, y = simple_linear_data
    reg = LinearRegression(x, y, limits=(-5.0, 5.0))
    assert reg.limits == (-5.0, 5.0)

    with pytest.raises(TypeError):
        LinearRegression(x, y, limits=(1.0,))  # type: ignore


def test_fit_basic_properties(simple_linear_data):
    x, y = simple_linear_data
    reg = LinearRegression(x, y).fit()

    assert np.isfinite(reg.slope)
    assert np.isfinite(reg.intercept)
    assert np.isfinite(reg.stderr)
    assert np.isfinite(reg.intercept_stderr)
    assert -1.0 <= reg.rvalue <= 1.0


def test_fit_recovers_parameters(simple_linear_data):
    x, y = simple_linear_data
    reg = LinearRegression(x, y).fit()

    assert np.isclose(reg.slope, 3.0, atol=0.1)
    assert np.isclose(reg.intercept, 2.0, atol=0.3)


def test_force_zero(simple_linear_data):
    x, y = simple_linear_data
    reg = LinearRegression(x, y, force_zero=True).fit()

    assert reg.intercept == 0.0
    assert reg.intercept_stderr == 0.0
    assert np.isclose(reg.slope, 3.0, atol=0.3)


def test_predict_y(simple_linear_data):
    x, y = simple_linear_data
    reg = LinearRegression(x, y).fit()

    y_pred = reg.predict_y([0.0, 1.0])
    assert np.allclose(y_pred, reg.slope * np.array([0.0, 1.0]) + reg.intercept)


def test_predict_x(simple_linear_data):
    x, y = simple_linear_data
    reg = LinearRegression(x, y).fit()

    y_test = np.array([2.0, 5.0])
    x_pred = reg.predict_x(y_test)

    assert np.allclose(reg.predict_y(x_pred), y_test)


def test_predict_raises_if_not_fitted():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    reg = LinearRegression(x, y)

    with pytest.raises(NotFittedError):
        reg.predict_y(1.0)

    with pytest.raises(NotFittedError):
        reg.predict_x(1.0)

    with pytest.raises(NotFittedError):
        reg.to_dataframe()


def test_iter_protocol(simple_linear_data):
    x, y = simple_linear_data
    reg = LinearRegression(x, y).fit()

    values = list(reg)
    assert len(values) == 5
    assert values[0] == reg.slope
    assert values[1] == reg.intercept


def test_str_and_repr(simple_linear_data):
    x, y = simple_linear_data
    reg = LinearRegression(x, y).fit()

    s = str(reg)
    r = repr(reg)

    assert "LinearRegression" in s
    assert s == r


def test_to_dataframe(simple_linear_data):
    x, y = simple_linear_data
    reg = LinearRegression(x, y).fit()

    df = reg.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == ["x", "u_x"]
    assert "slope" in df.columns
    assert "intercept" in df.columns
    assert "r_squared" in df.columns
    assert "rvalue" in df.columns

    assert np.isclose(pd.to_numeric(df.loc["x", "r_squared"]), reg.rvalue**2)
