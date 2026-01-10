import pytest
import numpy as np
from curve_fit import CurveFit
from exceptions import NotFittedError


def linear_func(x, a, b):
    return a * x + b


def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c


class TestCurveFit:
    def test_init_valid_data(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([2, 4, 6, 8])
        cf = CurveFit(linear_func, x, y)
        assert np.array_equal(cf.x, x)
        assert np.array_equal(cf.y, y)
        assert cf.fun == linear_func
        assert cf.limits == (1.0, 4.0)
        assert cf.p0 is None
        assert cf.bounds is None
        assert not cf.fitted
        with pytest.raises(NotFittedError):
            _ = cf.params
        with pytest.raises(NotFittedError):
            _ = cf.covariance
        with pytest.raises(NotFittedError):
            _ = cf.stderr

    def test_init_with_limits(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([2, 4, 6, 8])
        limits = (0.5, 5.0)
        cf = CurveFit(linear_func, x, y, limits=limits)
        assert cf.limits == limits

    def test_init_invalid_limits_wrong_length(self):
        x = np.array([1, 2, 3])
        y = np.array([2, 4, 6])
        with pytest.raises(
            ValueError,
            match="Limits must be a sequence of two numeric values",
        ):
            CurveFit(linear_func, x, y, limits=(0, 1, 2))

    def test_init_invalid_limits_not_numeric(self):
        x = np.array([1, 2, 3])
        y = np.array([2, 4, 6])
        with pytest.raises(
            ValueError,
            match="Limits must be a sequence of two numeric values",
        ):
            CurveFit(linear_func, x, y, limits=("a", "b"))

    def test_init_different_lengths(self):
        x = np.array([1, 2, 3])
        y = np.array([2, 4])
        with pytest.raises(ValueError, match="Input arrays must have the same length"):
            CurveFit(linear_func, x, y)

    def test_init_multidimensional_arrays(self):
        x = np.array([[1, 2], [3, 4]])
        y = np.array([2, 4])
        with pytest.raises(ValueError, match="Input arrays must be one-dimensional"):
            CurveFit(linear_func, x, y)

    def test_fit_linear(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([2, 4, 6, 8])
        cf = CurveFit(linear_func, x, y)
        cf.fit()
        assert cf.fitted
        assert cf.params is not None
        assert cf.covariance is not None
        # For linear y = 2x, params should be [2, 0] approximately
        assert np.allclose(cf.params, [2.0, 0.0], atol=1e-10)

    def test_fit_with_p0(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([3, 5, 7, 9])  # y = 2x + 1
        p0 = [1, 1]
        cf = CurveFit(linear_func, x, y, p0=p0)
        cf.fit()
        assert cf.fitted
        assert np.allclose(cf.params, [2.0, 1.0], atol=1e-10)

    def test_fit_with_bounds(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([2, 4, 6, 8])
        bounds = ([1, -1], [3, 1])
        cf = CurveFit(linear_func, x, y, bounds=bounds)
        cf.fit()
        assert cf.fitted
        assert 1 <= cf.params[0] <= 3
        assert -1 <= cf.params[1] <= 1

    def test_fit_quadratic(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 4, 9, 16])  # y = x^2
        cf = CurveFit(quadratic_func, x, y)
        cf.fit()
        assert cf.fitted
        assert np.allclose(cf.params, [1.0, 0.0, 0.0], atol=1e-10)

    def test_predict_y_valid(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([2, 4, 6, 8])
        cf = CurveFit(linear_func, x, y)
        cf.fit()
        x_pred = np.array([1.5, 2.5])
        y_pred = cf.predict_y(x_pred)
        expected = np.array([3.0, 5.0])  # Dla y = 2x
        assert np.allclose(y_pred, expected, atol=1e-10)

    def test_predict_y_not_fitted(self):
        x = np.array([1, 2, 3])
        y = np.array([2, 4, 6])
        cf = CurveFit(linear_func, x, y)
        with pytest.raises(NotFittedError):
            cf.predict_y([1.5])

    def test_predict_x_valid(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([2, 4, 6, 8])
        cf = CurveFit(linear_func, x, y)
        cf.fit()
        y_pred = np.array([3.0, 5.0])
        x_pred = cf.predict_x(y_pred)
        expected = np.array([1.5, 2.5])  # Dla y = 2x, x = y/2
        assert np.allclose(
            x_pred, expected, atol=1e-2
        )  # Tolerancja dla numerycznego rozwiązywania

    def test_predict_x_not_fitted(self):
        x = np.array([1, 2, 3])
        y = np.array([2, 4, 6])
        cf = CurveFit(linear_func, x, y)
        with pytest.raises(NotFittedError):
            cf.predict_x([3.0])

    def test_r2_property(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([2, 4, 6, 8])
        cf = CurveFit(linear_func, x, y)
        cf.fit()
        assert cf.r2 == pytest.approx(1.0, abs=1e-10)  # Doskonałe dopasowanie

    def test_r2_not_fitted(self):
        x = np.array([1, 2, 3])
        y = np.array([2, 4, 6])
        cf = CurveFit(linear_func, x, y)
        with pytest.raises(NotFittedError):
            _ = cf.r2

    def test_covariance_property(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([2, 4, 6, 8])
        cf = CurveFit(linear_func, x, y)
        cf.fit()
        assert cf.covariance.shape == (2, 2)  # Dla 2 parametrów

    def test_covariance_not_fitted(self):
        x = np.array([1, 2, 3])
        y = np.array([2, 4, 6])
        cf = CurveFit(linear_func, x, y)
        with pytest.raises(NotFittedError):
            _ = cf.covariance

    def test_iter_fitted(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([2, 4, 6, 8])
        cf = CurveFit(linear_func, x, y)
        cf.fit()
        params, stderr, r2 = cf.params, cf.stderr, cf.r2
        iter_values = list(cf)
        expected = list(params) + list(stderr) + [r2]
        assert iter_values == pytest.approx(expected, abs=1e-10)

    def test_iter_not_fitted(self):
        x = np.array([1, 2, 3])
        y = np.array([2, 4, 6])
        cf = CurveFit(linear_func, x, y)
        with pytest.raises(NotFittedError):
            list(cf)

    def test_str_fitted(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([2, 4, 6, 8])
        cf = CurveFit(linear_func, x, y)
        cf.fit()
        str_repr = str(cf)
        assert "params=" in str_repr
        assert "stderr=" in str_repr
        assert "r2=" in str_repr

    def test_str_not_fitted(self):
        x = np.array([1, 2, 3])
        y = np.array([2, 4, 6])
        cf = CurveFit(linear_func, x, y)
        str_repr = str(cf)
        assert "not fitted" in str_repr

    def test_init_limits_none(self):
        x = np.array([1, 2, 3])
        y = np.array([2, 4, 6])
        cf = CurveFit(linear_func, x, y)
        assert cf.limits == (1.0, 3.0)  # Min i max z x
