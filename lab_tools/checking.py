import numpy as np


def check_if_equal_extended_unc(
    x1: float | int, x2: float | int, u_x1: float | int, u_x2: float | int
) -> None:
    """Prints whether `x1` and `x2` are equal within the extended uncertainty bounds, using the formula `|x1 - x2| <= k * sqrt(u_x1^2 + u_x2^2)`.

    :param x1: First value
    :type x1: float | int
    :param x2: Second value
    :type x2: float | int
    :param u_x1: Uncertainty of the first value
    :type u_x1: float | int
    :param u_x2: Uncertainty of the second value
    :type u_x2: float | int
    :raises TypeError: if parameters are not of type int or float
    """
    # Ensure all inputs are either int or float
    for var, name in zip([x1, x2, u_x1, u_x2], ["x1", "x2", "u_x1", "u_x2"]):
        if not isinstance(var, (int, float)):
            raise TypeError(f"{name} must be float or int, but got {type(var)}")

    is_equal = False
    k: int = 1
    while not is_equal:
        k += 1
        U_diff: float = k * np.sqrt(u_x1**2 + u_x2**2)
        diff: float = np.abs(x1 - x2)
        if diff <= U_diff:
            is_equal = True

    print(
        f"Equal within the extended uncertainty bounds ({'only for ' if k > 2 else ''}k = {k})."
    )
