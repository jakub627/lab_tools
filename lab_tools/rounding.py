from typing import Union
import numpy as np


def round_to_2(
    x: Union[int, float, list[Union[int, float]]],
    u_x: Union[int, float, list[Union[int, float]], None] = None,
) -> Union[int, float, list[Union[int, float]]]:
    """
    Rounds values to two significant figures or to a specified uncertainty

    :param x: A number or an array of numbers to be rounded
    :param u_x: The precision to which the values of `x` should be rounded, defaults to `None`
    :raises TypeError: Type of `x` is not `int`, `float`, or `list[int | float]`
    :raises TypeError: Type of `u_x` is not `int`, `float`, or `list[int | float]`
    :raises ValueError: `x` and `u_x` are `list[int | float]` of different lengths
    :raises ValueError: `u_x` is an array and `x` is a number
    :raises ValueError: `u_x` has negative elements
    :raises ValueError: `u_x` is negative
    :raises ValueError: `u_x` and `x` have different lengths
    :return: Rounded values to two significant figures or to the uncertainty (if provided)
    """
    # Checking the type of x
    if not isinstance(x, (int, float)) and not (
        isinstance(x, list) and all(isinstance(i, (float, int)) for i in x)
    ):
        # Determine the type for the error message
        if isinstance(x, list):
            error_type: str = (
                f"list[{' | '.join([*set([type(i).__name__ for i in x])])}]"
            )
        else:
            error_type: str = type(x).__name__
        raise TypeError(
            f"x must be either an int, a float, or a list of int or float, but got {error_type}"
        )
    # Checking the type of u_x
    if u_x is not None:
        if not isinstance(u_x, (int, float)) and not (
            isinstance(u_x, list) and all(isinstance(i, (float, int)) for i in u_x)
        ):
            # Determine the type for the error message
            if isinstance(u_x, list):
                error_type: str = (
                    f"list[{' | '.join([*set([type(i).__name__ for i in u_x])])}]"
                )
            else:
                error_type: str = type(u_x).__name__
            raise TypeError(
                f"u_x must be either an int, a float, or a list of int or float, but got {error_type}"
            )

        # Checking if x is a list and u_x is a number
        if isinstance(u_x, list) and isinstance(x, (int, float)):
            raise ValueError("u_x cannot be a list while x is an int or a float")

        # Checking if u_x is greater than or equal to zero
        np_u_x = np.array(u_x)
        if isinstance(u_x, list):
            if not np.all(np_u_x >= 0):
                raise ValueError(
                    "All elements in u_x must be greater than or equal to zero"
                )
        else:  # u_x is int or float
            if u_x < 0:
                raise ValueError("u_x cannot be less than zero")
    # Checking length, if both are lists
    if isinstance(x, list) and isinstance(u_x, list):
        if len(x) != len(u_x):
            raise ValueError(
                f"x and u_x must have the same size, but got {len(x)} and {len(u_x)}"
            )

    def round_index(
        x: Union[int, float, list[Union[int, float]]]
    ) -> Union[int, list[int]]:
        """
        Returns the index of the second significant digit of a number (for absolute values greater than 1 - negative, for values less than 1 - positive)

        :param x: The number for which we are looking for the index of the second significant digit
        :return: The index of the second significant digit
        """
        np_x = np.array(x)
        np_x = (-np.floor(np.log10(np.abs(np_x / 10)))).astype(int)
        return np_x.tolist() if isinstance(x, list) else int(np_x)

    def round_non_zero_elements(
        x: Union[int, float, list[Union[int, float]]],
        round_indices: Union[int, list[int]],
    ) -> Union[float, list[Union[int, float]]]:
        """
        Rounds a number to two significant digits

        :param x: The number we want to round
        :param round_indices: Indices of the second significant digit
        :raises ValueError: An unexpected error occurred
        :return: The number rounded to two significant digits
        """
        if isinstance(x, list) and isinstance(round_indices, list):
            for i in range(len(x)):
                x[i] = round(x[i], round_indices[i])
        elif isinstance(x, (int, float)) and isinstance(round_indices, int):
            x = round(x, round_indices)
        elif isinstance(x, list) and isinstance(round_indices, int):
            for i in range(len(x)):
                x[i] = round(x[i], round_indices)
        else:
            raise ValueError("Unexpected error has occurred")

        return x

    u_x = (
        None if isinstance(u_x, (int, float)) and u_x == 0 else u_x
    )  # Setting u_x to None if uncertainty is equal to zero

    if isinstance(x, (int, float)) and x == 0:
        return 0

    if u_x is None:  # No u_x
        if isinstance(x, list):  # x is list

            # Indices of elements not equal to zero
            non_zero_indices = [i for i, value in enumerate(x) if value != 0]

            # Select non-zero elements
            x_non_zero = [x[i] for i in non_zero_indices]

            round_indices = round_index(x_non_zero)

            # Round the non-zero elements based on some rounding indices
            x_rounded_non_zero = round_non_zero_elements(x_non_zero, round_indices)

            # Assign the rounded non-zero elements back to their original positions in x
            for i, rounded_value in zip(non_zero_indices, x_rounded_non_zero):  # type: ignore
                x[i] = rounded_value

        else:  # x is a number
            x = round_non_zero_elements(x, round_index(x))
    else:  # There is u_x
        if isinstance(x, list) and isinstance(u_x, list):
            u_x_non_zero_indices = [i for i, value in enumerate(u_x) if value != 0]
            u_x_zero_indices = [i for i, value in enumerate(u_x) if value == 0]

            # Rounding x-es which u_x is equal to zero
            for i in u_x_zero_indices:
                if x[i] != 0:
                    x[i] = round_non_zero_elements(x[i], round_index(x[i]))  # type: ignore
            for i in u_x_non_zero_indices:
                if x[i] != 0:
                    x[i] = round_non_zero_elements(x[i], round_index(u_x[i]))  # type: ignore

        else:
            x = round_non_zero_elements(x, round_index(u_x))

    return x


def rounded_string(
    x: float,
    u_x: float = 0,
    s_x: str = "x",
    unit_x: str = "",
    scale_x: int = 0,
    scale_u_x: None | int = None,
    is_unc: bool = True,
    t: bool | int = False,
    n: bool | int = False,
) -> str:
    """
    Returns a string of a value along with its uncertainty, rounded to two significant digits and their units.
    The decimal points of the values are aligned.

    Parameters
    ----------
    `x` : `float`
        Measured value to be printed.
    `u_x` : `float`, optional
        Uncertainty of the measured value, by default 0
    `s_x` : `str`, optional
        Name of the measured value, by default `"x"`
    `unit_x` : `str`, optional
        Unit of the measured value, by default `""`
    `scale_x` : `int`, optional
        Scale - the power of 10 by which the measured value will be multiplied, by default `0`
    `scale_u_x` : `None | int`, optional
        Scale - the power of 10 by which the uncertainty will be multiplied, by default `None`
    `is_unc` : `bool`, optional
        Wether to take uncertainty into account when returning string, by default `True`
    `t` : `bool | int`, optional
        Inserts a tabulation before printing, by default `False`
    `n` : `bool | int`, optional
        Inserts a new line before printing, by default `False`

    Returns
    -------
    `str`
        The formatted string with the measured value and its uncertainty.

    Raises
    ------
    `ValueError`
        If the exponent is not a valid SI exponent.
    """

    def exponent_to_prefix(exponent: int) -> str:
        prefix_map = {
            -24: "y",  # yocto
            -21: "z",  # zepto
            -18: "a",  # atto
            -15: "f",  # femto
            -12: "p",  # pico
            -9: "n",  # nano
            -6: "u",  # micro
            -3: "m",  # milli
            -2: "c",  # centi
            -1: "d",  # deci
            0: "",  # no prefix
            1: "da",  # deca
            2: "h",  # hecto
            3: "k",  # kilo
            6: "M",  # mega
            9: "G",  # giga
            12: "T",  # tera
            15: "P",  # peta
            18: "E",  # exa
            21: "Z",  # zetta
            24: "Y",  # yotta
        }
        if exponent in prefix_map:
            return prefix_map[exponent]
        else:
            raise ValueError(f"Exponent '{exponent}' is not a valid SI exponent.")

    scale_u_x = scale_x if not scale_u_x else scale_u_x
    s_u_x = None if (not u_x or not is_unc) else "u_" + s_x

    x_rounded = round_to_2(x * 10**-scale_x, u_x * 10**-scale_x)
    u_x_rounded = round_to_2(u_x * 10**-scale_u_x)

    x_rounded_str = f"{x_rounded:f}".rstrip("0").rstrip(".")
    u_x_rounded_str = f"{u_x_rounded:f}".rstrip("0").rstrip(".")

    if scale_x == scale_u_x:
        rounding_index = len(x_rounded_str.split(".")[1]) if "." in x_rounded_str else 0
        x_rounded_str = f"{x_rounded:.{rounding_index}f}"
        u_x_rounded_str = f"{u_x_rounded:.{rounding_index}f}"

    max_val_len = max(len(str(int(x_rounded))), len(str(int(x_rounded)))) + len(u_x_rounded_str) - 1 if scale_x == scale_u_x and u_x else 0  # type: ignore
    max_str_len = len(s_u_x) if s_u_x else 0

    newline = "\n" if n else ""
    tab = "\t" if t else ""

    prefix_x = exponent_to_prefix(scale_x)
    prefix_u_x = exponent_to_prefix(scale_u_x)

    x_str = f"{newline}{tab}{s_x:>{max_str_len}} = {x_rounded_str:>{max_val_len}}{f' [{prefix_x}{unit_x}]' if unit_x else ''}"
    u_x_str = (
        f"\n{tab}{s_u_x:>{max_str_len}} = {u_x_rounded_str:>{max_val_len}}{f' [{prefix_u_x}{unit_x}]' if unit_x else ''}"
        if u_x
        else ""
    )

    return x_str + u_x_str
