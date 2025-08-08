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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_bg(
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    *,
    xformat: int | None = None,
    yformat: int | None = None,
    xscale: str | None = None,
    yscale: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    margins: tuple[float, float] = (0.05, 0.05),
    isLegend: bool | None = None,
    isGrid: bool | None = None,
    facecolor: str | None = None,
    grid_style: str = "--",
    grid_alpha: float = 0.6,
    fontsize_x: int = 12,
    fontsize_y: int = 12,
    fontsize_title: int = 14,
    fontsize_legend: int = 12,
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
    title : str, optional
        Title of the plot, by default ""
    xformat : int, optional
        Exponent threshold for scientific notation on the x-axis, by default 0
    yformat : int, optional
        Exponent threshold for scientific notation on the y-axis, by default 0
    xscale : str, optional
        Scale type for the x-axis ('linear', 'log', etc.), by default "linear"
    yscale : str, optional
        Scale type for the y-axis ('linear', 'log', etc.), by default "linear"
    xlim : tuple[float, float] | None, optional
        Limits for the x-axis, by default None
    ylim : tuple[float, float] | None, optional
        Limits for the y-axis, by default None
    margins : tuple[float, float], optional
        Relative margins to apply if limits are set manually, by default (0.05, 0.05)
    isLegend : bool, optional
        Whether to display the legend, by default False
    isGrid : bool, optional
        Whether to display a grid, by default True
    facecolor : str, optional
        Background color of the plot area, by default "#f4f4f4"
    grid_style : str, optional
        Line style for the grid, by default "--"
    grid_alpha : float, optional
        Transparency for the grid lines, by default 0.6
    fontsize_x : int, optional
        Font size for the x-axis label, by default 12
    fontsize_y : int, optional
        Font size for the y-axis label, by default 12
    fontsize_title : int, optional
        Font size for the plot title, by default 14
    fontsize_legend : int, optional
        Font size for the legend text, by default 12
    ax : Axes | None, optional
        A Matplotlib Axes object to apply formatting to, by default None

    Raises
    ------
    TypeError
        If xlim or ylim is provided but not a tuple of two floats
    """
    if ax is None:
        ax = plt.gca()

    # Scaling
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)

    # Background
    if facecolor is not None:
        ax.set_facecolor(facecolor)

    # Labels and title
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize_x)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize_y)
    if title is not None:
        ax.set_title(title, fontsize=fontsize_title)

    # Scientific notation
    if xformat is not None:
        ax.ticklabel_format(style="sci", axis="x", scilimits=(xformat, xformat))
    if yformat is not None:
        ax.ticklabel_format(style="sci", axis="y", scilimits=(yformat, yformat))

    # Limits and margins
    if xlim is not None:
        if not (
            isinstance(xlim, tuple)
            and len(xlim) == 2
            and all(isinstance(v, (int, float)) for v in xlim)
        ):
            raise TypeError("xlim must be a tuple of two floats.")
        dx = abs(xlim[1] - xlim[0])
        ax.set_xlim(xlim[0] - margins[0] * dx, xlim[1] + margins[0] * dx)
    else:
        ax.margins(x=margins[0])

    if ylim is not None:
        if not (
            isinstance(ylim, tuple)
            and len(ylim) == 2
            and all(isinstance(v, (int, float)) for v in ylim)
        ):
            raise TypeError("ylim must be a tuple of two floats.")
        dy = abs(ylim[1] - ylim[0])
        ax.set_ylim(ylim[0] - margins[1] * dy, ylim[1] + margins[1] * dy)
    else:
        ax.margins(y=margins[1])

    # Legend
    if isLegend is not None:
        ax.legend(
            fontsize=fontsize_legend,
            loc="best",
            frameon=True,
            fancybox=True,
            edgecolor="black",
        )

    # Grid
    if isGrid is not None:
        ax.grid(True, linestyle=grid_style, alpha=grid_alpha)


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
