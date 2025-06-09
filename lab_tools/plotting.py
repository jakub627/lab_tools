import array
from turtle import bgcolor, color
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure
from matplotlib.axes import Axes
from numpy._typing._array_like import NDArray
from numpy.typing import ArrayLike
from scipy.interpolate import make_interp_spline


def plot_bg(
    xlabel: str = "",
    ylabel: str = "",
    xformat: int = 0,
    yformat: int = 0,
    isLegend: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    facecolor: str = "#f4f4f4",
    fontsize_x: int = 12,
    fontsize_y: int = 12,
    fontsize_legend: int = 12,
    isGrid: bool = True,
    ax: Axes | None = None,
) -> None:
    """
    Configure and style a Matplotlib Axes object for plotting.

    Parameters
    ----------
    xlabel : str, optional
        Label for the x-axis, by default ""
    ylabel : str, optional
        Label for the y-axis, by default ""
    xformat : int, optional
        Exponent threshold for scientific notation on the x-axis, by default 0
    yformat : int, optional
        Exponent threshold for scientific notation on the y-axis, by default 0
    isLegend : bool, optional
        Whether to display the legend, by default False
    xlim : tuple[float, float] | None, optional
        Limits for the x-axis, by default None
    ylim : tuple[float, float] | None, optional
        Limits for the y-axis, by default None
    facecolor : str, optional
        Background color of the plot area, by default "#f4f4f4"
    fontsize_x : int, optional
        Font size for the x-axis label, by default 12
    fontsize_y : int, optional
        Font size for the y-axis label, by default 12
    fontsize_legend : int, optional
        Font size for the legend text, by default 12
    isGrid : bool, optional
        Whether to display a grid, by default True
    ax : Axes | None, optional
        A Matplotlib Axes object to apply formatting to, by default None

    Raises
    ------
    TypeError
        If xlim is provided but is not a tuple of two floats
    TypeError
        If ylim is provided but is not a tuple of two floats
    """
    if ax is None:
        ax = plt.gca()

    # Set background color
    ax.set_facecolor(facecolor)

    # Axis labels
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize_x)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize_y)

    # Scientific notation
    if xformat:
        ax.ticklabel_format(style="sci", axis="x", scilimits=(xformat, xformat))
    if yformat:
        ax.ticklabel_format(style="sci", axis="y", scilimits=(yformat, yformat))

    # Limits
    if xlim is not None:
        if isinstance(xlim, tuple):
            ax.set_xlim(xlim)
        else:
            raise TypeError(f"xlim must be a tuple, got {type(xlim)}")

    if ylim is not None:
        if isinstance(ylim, tuple):
            ax.set_ylim(ylim)
        else:
            raise TypeError(f"ylim must be a tuple, got {type(ylim)}")

    # Legend
    if isLegend:
        ax.legend(
            fontsize=fontsize_legend,
            loc="best",
            frameon=True,
            fancybox=True,
            edgecolor="black",
        )

    # Grid
    ax.grid(isGrid, linestyle="--", alpha=0.6 if isGrid else 0)


def plot_errorbar(
    x: ArrayLike,
    y: ArrayLike,
    yerr: ArrayLike | None = None,
    xerr: ArrayLike | None = None,
    color: str = "#FF0000",
    label: str = "Values",
    fmt: str = ".",
    capsize: int = 4,
    markersize: int | None = None,
) -> None:

    plt.errorbar(
        x,
        y,
        yerr,
        xerr,
        fmt=fmt,
        markersize=markersize,
        capsize=capsize,
        label=label,
        c=color,
    )


def plot_plot(
    x: ArrayLike,
    y: ArrayLike,
    color: str = "#FF0000",
    label: str = "Values",
    fmt: str = "",
    markersize: int | None = None,
) -> None:

    plt.plot(
        x,
        y,
        fmt=fmt,
        markersize=markersize,
        label=label,
        c=color,
    )


def plot_scatter(
    x: ArrayLike,
    y: ArrayLike,
    color: str = "#FF0000",
    label: str = "Values",
    fmt: str = "",
    markersize: int | None = None,
) -> None:

    plt.scatter(
        x,
        y,
        fmt=fmt,
        markersize=markersize,
        label=label,
        c=color,
    )


def interpolate(
    x: ArrayLike, y: ArrayLike, k: int = 3, npoints: int = 300
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    Calculates the points for interpolated plot.

    :param ArrayLike x: X values; an array-like sequence (e.g., list, tuple, or np.ndarray) of floats
    :param ArrayLike y: Y values; an array-like sequence (e.g., list, tuple, or np.ndarray) of floats
    :param int k: B-spline degree. Default is cubic (`k = 3`)
    :param int npoints: Number of generated points, defaults to 300
    :return Tuple[NDArray[np.float64], NDArray[np.float64]]: `x` and `y` values of interpolated points as 1D NumPy arrays of floats
    """
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)

    x_new = np.linspace(np.min(x_array), np.max(x_array), npoints)

    spl = make_interp_spline(x_array, y_array, k)
    y_new = spl(x_new)

    return x_new, y_new
