import array
from turtle import bgcolor, color
from typing import Any, List, Tuple
from matplotlib import figure
import matplotlib.pyplot as plt
import numpy as np
from numpy._typing._array_like import NDArray
from scipy.interpolate import make_interp_spline
from numpy.typing import ArrayLike


def plot_bg(
    xlabel: str = "",
    ylabel: str = "",
    xformat: int = 0,
    yformat: int = 0,
    isLegend: bool = False,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
    facecolor: str = "#f4f4f4",
    isGrid: bool = True,
) -> None:
    """Creates the figure for a plot

    :param xlabel: Label for the x-axis, defaults to ""
    :type xlabel: str, optional
    :param ylabel: Label for the y-axis, defaults to ""
    :type ylabel: str, optional
    :param xformat: Order of magnitude for the x-axis, defaults to 0
    :type xformat: int, optional
    :param yformat: Order of magnitude for the y-axis, defaults to 0
    :type yformat: int, optional
    :param xlim: Limits for the x-axis as a tuple, defaults to None
    :type xlim: tuple | None, optional
    :param ylim: Limits for the y-axis as a tuple, defaults to None
    :type ylim: tuple | None, optional
    :param isLegend: Whether to display the legend, defaults to False
    :type isLegend: bool, optional
    :param facecolor: Background color for the plot, defaults to "#f4f4f4"
    :type facecolor: str, optional
    :param isGrid: Whether to show the grid, defaults to True
    :type isGrid: bool, optional
    :raises TypeError: If `xlim` is not a tuple
    :raises TypeError: If `ylim` is not a tuple
    """
    plt.grid(True, linestyle="--", alpha=0.6)
    if xformat:
        plt.ticklabel_format(style="sci", axis="x", scilimits=(xformat, xformat))
    if yformat:
        plt.ticklabel_format(style="sci", axis="y", scilimits=(yformat, yformat))
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)

    if xlim:
        if isinstance(xlim, tuple):
            plt.xlim(xlim)
        else:
            raise TypeError(f"xlim must be of type tuple, got {type(xlim)}")

    if ylim:
        if isinstance(ylim, tuple):
            plt.ylim(ylim)
        else:
            raise TypeError(f"ylim must be of type tuple, got {type(ylim)}")

    if isLegend:
        plt.legend(fontsize=10)

    plt.grid(isGrid, linestyle="--", alpha=0.6 if isGrid else 0)

    ax = plt.gca()
    if facecolor:
        ax.set_facecolor(facecolor)


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
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
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
