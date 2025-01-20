import sympy as sp
import numpy as np


def evaluate_with_uncertainty(
    formula: sp.Basic,
    variables: list[sp.Symbol],
    values: list[float],
    uncertainties: list[float],
) -> tuple[float, float]:
    """
    Calculates the value of the function and the uncertainty of the result according to the propagation of uncertainty law.

    Parameters
    ----------
    `formula` : `sp.Mul`
        The formula of the function `f(x1, x2, ..., xn)`
    `variables` : `list[sp.Symbol]`
        List of variables `[x1, x2, ..., xn]`

    `values` : `list[float]`
        Values of the variables `[x1_val, x2_val, ..., xn_val]`
    `uncertainties` : `list[float]`
        Uncertainties of the variables `[u_x1, u_x2, ..., u_xn]`


    Returns
    -------
    `tuple[float,float]`
        The value of the function and its uncertainty

    Raises
    ------
    `TypeError`
        If the inputs are not of the expected types.
    `ValueError`
        If the lengths of variables, values, and uncertainties do not match.
    """

    # Type checking
    if not isinstance(formula, sp.Basic):
        raise TypeError(
            f"Formula must be a sympy expression (sp.Basic), but got {type(formula)}"
        )
    if not all(isinstance(var, sp.Symbol) for var in variables):
        invalid_types = [
            type(var) for var in variables if not isinstance(var, sp.Symbol)
        ]
        raise TypeError(
            f"All variables must be of type sympy.Symbol, but got {invalid_types}"
        )
    if not all(isinstance(val, (int, float)) for val in values):
        invalid_types = [
            type(val) for val in values if not isinstance(val, (int, float))
        ]
        raise TypeError(
            f"All values must be integers or floats, but got {invalid_types}"
        )
    if not all(isinstance(unc, (int, float)) for unc in uncertainties):
        invalid_types = [
            type(unc) for unc in uncertainties if not isinstance(unc, (int, float))
        ]
        raise TypeError(
            f"All uncertainties must be integers or floats, but got {invalid_types}"
        )

    # Length checking
    if not (len(variables) == len(values) == len(uncertainties)):
        lengths = {
            "variables": len(variables),
            "values": len(values),
            "uncertainties": len(uncertainties),
        }
        raise ValueError(
            f"The lengths of variables, values, and uncertainties must be equal, but got {lengths}."
        )

    # Calculate the value of the function
    value = formula.subs(dict(zip(variables, values)))

    # Calculate partial derivatives and uncertainties
    partial_derivatives = [sp.diff(formula, var) for var in variables]
    squared_uncertainties = []

    for i, deriv in enumerate(partial_derivatives):
        deriv_value = deriv.subs(dict(zip(variables, values)))
        squared_uncertainties.append((float(sp.N(deriv_value)) * uncertainties[i]) ** 2)

    # Calculate total uncertainty
    uncertainty = np.sqrt(sum(squared_uncertainties))

    return sp.N(value), uncertainty
