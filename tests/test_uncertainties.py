import pytest
import sympy as sp
from lab_tools.uncertainties import evaluate_with_uncertainty


def test_valid_case():
    """
    Tests the function with a valid two-variable formula.
    Ensures the result matches the expected value and uncertainty.
    """
    x, y = sp.symbols("x y")
    formula = x * y
    variables = [x, y]
    values = [3.0, 4.0]
    uncertainties = [0.1, 0.2]
    result = evaluate_with_uncertainty(formula, variables, values, uncertainties)
    assert result == pytest.approx((12.0, 0.7211), rel=1e-3)


def test_single_variable():
    """
    Tests the function with a single-variable formula.
    Ensures it correctly computes the value and uncertainty.
    """
    x = sp.Symbol("x")
    formula = x**2
    variables = [x]
    values = [2.0]
    uncertainties = [0.1]
    result = evaluate_with_uncertainty(formula, variables, values, uncertainties)
    assert result == pytest.approx((4.0, 0.4), rel=1e-3)


def test_multiple_variables():
    """
    Tests the function with a formula containing three variables.
    Ensures it sums uncertainties from all variables correctly.
    """
    x, y, z = sp.symbols("x y z")
    formula = x + y + z
    variables = [x, y, z]
    values = [1.0, 2.0, 3.0]
    uncertainties = [0.1, 0.2, 0.3]
    result = evaluate_with_uncertainty(formula, variables, values, uncertainties)
    assert result == pytest.approx((6.0, 0.374), rel=1e-3)


def test_type_error_formula():
    """
    Tests that passing a non-sympy formula raises a TypeError.
    """
    with pytest.raises(TypeError, match="Formula must be a sympy expression"):
        evaluate_with_uncertainty("x * y", [sp.Symbol("x"), sp.Symbol("y")], [1.0, 2.0], [0.1, 0.2])  # type: ignore


def test_type_error_variables():
    """
    Tests that passing non-sympy.Symbol variables raises a TypeError.
    """
    with pytest.raises(TypeError, match="All variables must be of type sympy.Symbol"):
        evaluate_with_uncertainty(sp.Symbol("x") * sp.Symbol("y"), ["x", "y"], [1.0, 2.0], [0.1, 0.2])  # type: ignore


def test_type_error_values():
    """
    Tests that passing non-numeric values in the values list raises a TypeError.
    """
    x, y = sp.symbols("x y")
    with pytest.raises(TypeError, match="All values must be integers or floats"):
        evaluate_with_uncertainty(x * y, [x, y], [1.0, "2.0"], [0.1, 0.2])  # type: ignore


def test_type_error_uncertainties():
    """
    Tests that passing non-numeric values in the uncertainties list raises a TypeError.
    """
    x, y = sp.symbols("x y")
    with pytest.raises(TypeError, match="All uncertainties must be integers or floats"):
        evaluate_with_uncertainty(x * y, [x, y], [1.0, 2.0], [0.1, "0.2"])  # type: ignore


def test_length_mismatch():
    """
    Tests that mismatched lengths of variables, values, and uncertainties raise a ValueError.
    """
    x, y = sp.symbols("x y")
    with pytest.raises(
        ValueError,
        match="The lengths of variables, values, and uncertainties must be equal",
    ):
        evaluate_with_uncertainty(x * y, [x, y], [1.0], [0.1, 0.2])


def test_zero_uncertainty():
    """
    Tests the case where all uncertainties are zero.
    Ensures the result has zero uncertainty for the output value.
    """
    x, y = sp.symbols("x y")
    formula = x * y
    variables = [x, y]
    values = [3.0, 4.0]
    uncertainties = [0.0, 0.0]
    result = evaluate_with_uncertainty(formula, variables, values, uncertainties)
    assert result == pytest.approx((12.0, 0.0), rel=1e-3)
