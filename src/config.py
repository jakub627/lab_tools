from typing import Literal, cast

import pint
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pint import Quantity
from scipy.constants import physical_constants


class Formatters:

    @staticmethod
    def scalar_formatter(
        useOffset: bool | float | None = None,
        useMathText: bool | None = None,
        useLocale: bool | None = None,
        *,
        usetex: bool | None = None,
        set_scientific: bool = True,
        power_limits: tuple[int, int] = (-2, 3),
    ) -> ScalarFormatter:
        """
        Format tick values as a number.

        Parameters
        ----------
        useOffset : bool or float, default: :rc:`axes.formatter.useoffset`
            Whether to use offset notation. See `.set_useOffset`.
        useMathText : bool, default: :rc:`axes.formatter.use_mathtext`
            Whether to use fancy math formatting. See `.set_useMathText`.
        useLocale : bool, default: :rc:`axes.formatter.use_locale`.
            Whether to use locale settings for decimal sign and positive sign.
            See `.set_useLocale`.
        usetex : bool, default: :rc:`text.usetex`
            To enable/disable the use of TeX's math mode for rendering the
            numbers in the formatter.

            .. versionadded:: 3.10

        Notes
        -----
        In addition to the parameters above, the formatting of scientific vs.
        floating point representation can be configured via `.set_scientific`
        and `.set_powerlimits`).

        **Offset notation and scientific notation**

        Offset notation and scientific notation look quite similar at first sight.
        Both split some information from the formatted tick values and display it
        at the end of the axis.

        - The scientific notation splits up the order of magnitude, i.e. a
          multiplicative scaling factor, e.g. ``1e6``.

        - The offset notation separates an additive constant, e.g. ``+1e6``. The
          offset notation label is always prefixed with a ``+`` or ``-`` sign
          and is thus distinguishable from the order of magnitude label.

        The following plot with x limits ``1_000_000`` to ``1_000_010`` illustrates
        the different formatting. Note the labels at the right edge of the x axis.

        .. plot::

            lim = (1_000_000, 1_000_010)

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'hspace': 2})
            ax1.set(title='offset notation', xlim=lim)
            ax2.set(title='scientific notation', xlim=lim)
            ax2.xaxis.get_major_formatter().set_useOffset(False)
            ax3.set(title='floating-point notation', xlim=lim)
            ax3.xaxis.get_major_formatter().set_useOffset(False)
            ax3.xaxis.get_major_formatter().set_scientific(False)

        """
        formatter = ScalarFormatter(useOffset, useMathText, useLocale, usetex=usetex)
        formatter.set_scientific(set_scientific)
        formatter.set_powerlimits(power_limits)
        return formatter


class Locators:

    @staticmethod
    def MaxNLocator(
        nbins: int | Literal["auto"] | None = None, **kwargs
    ) -> MaxNLocator:
        """
        Parameters
        ----------
        nbins : int or 'auto', default: 10
            Maximum number of intervals; one less than max number of
            ticks.  If the string 'auto', the number of bins will be
            automatically determined based on the length of the axis.

        steps : array-like, optional
            Sequence of acceptable tick multiples, starting with 1 and
            ending with 10. For example, if ``steps=[1, 2, 4, 5, 10]``,
            ``20, 40, 60`` or ``0.4, 0.6, 0.8`` would be possible
            sets of ticks because they are multiples of 2.
            ``30, 60, 90`` would not be generated because 3 does not
            appear in this example list of steps.

        integer : bool, default: False
            If True, ticks will take only integer values, provided at least
            *min_n_ticks* integers are found within the view limits.

        symmetric : bool, default: False
            If True, autoscaling will result in a range symmetric about zero.

        prune : {'lower', 'upper', 'both', None}, default: None
            Remove the 'lower' tick, the 'upper' tick, or ticks on 'both' sides
            *if they fall exactly on an axis' edge* (this typically occurs when
            :rc:`axes.autolimit_mode` is 'round_numbers').  Removing such ticks
            is mostly useful for stacked or ganged plots, where the upper tick
            of an Axes overlaps with the lower tick of the axes above it.

        min_n_ticks : int, default: 2
            Relax *nbins* and *integer* constraints if necessary to obtain
            this minimum number of ticks.
        """
        return MaxNLocator(nbins, **kwargs)


class AxesUtils:
    @staticmethod
    def cax(ax: Axes) -> Axes:
        """
        Create and return a colorbar axes next to the given axes

        Parameters
        ----------
        ax : Axes
            The axes to create a colorbar axes for

        Returns
        -------
        Axes
            The colorbar axes
        """
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        return cax


class Norms:
    @staticmethod
    def log(
        vmin: float | None = None, vmax: float | None = None, clip: bool = False
    ) -> LogNorm:
        """
        Parameters
        ----------
        vmin, vmax : float or None
            Values within the range ``[vmin, vmax]`` from the input data will be
            linearly mapped to ``[0, 1]``. If either *vmin* or *vmax* is not
            provided, they default to the minimum and maximum values of the input,
            respectively.

        clip : bool, default: False
            Determines the behavior for mapping values outside the range
            ``[vmin, vmax]``.

            If clipping is off, values outside the range ``[vmin, vmax]`` are
            also transformed, resulting in values outside ``[0, 1]``.  This
            behavior is usually desirable, as colormaps can mark these *under*
            and *over* values with specific colors.

            If clipping is on, values below *vmin* are mapped to 0 and values
            above *vmax* are mapped to 1. Such values become indistinguishable
            from regular boundary values, which may cause misinterpretation of
            the data.

        Notes
        -----
        If ``vmin == vmax``, input data will be mapped to 0.
        """
        return LogNorm(vmin=vmin, vmax=vmax, clip=clip)

    @staticmethod
    def linear(
        vmin: float | None = None, vmax: float | None = None, clip: bool = False
    ) -> Normalize:
        """
        Parameters
        ----------
        vmin, vmax : float or None
            Values within the range ``[vmin, vmax]`` from the input data will be
            linearly mapped to ``[0, 1]``. If either *vmin* or *vmax* is not
            provided, they default to the minimum and maximum values of the input,
            respectively.

        clip : bool, default: False
            Determines the behavior for mapping values outside the range
            ``[vmin, vmax]``.

            If clipping is off, values outside the range ``[vmin, vmax]`` are
            also transformed, resulting in values outside ``[0, 1]``.  This
            behavior is usually desirable, as colormaps can mark these *under*
            and *over* values with specific colors.

            If clipping is on, values below *vmin* are mapped to 0 and values
            above *vmax* are mapped to 1. Such values become indistinguishable
            from regular boundary values, which may cause misinterpretation of
            the data.

        Notes
        -----
        If ``vmin == vmax``, input data will be mapped to 0.
        """
        return Normalize(vmin=vmin, vmax=vmax, clip=clip)


class SI:
    ureg: pint.UnitRegistry = pint.UnitRegistry()
    ureg.formatter.default_format = "~P"
    Q_ = ureg.Quantity

    k_B = cast(Quantity, Q_(*physical_constants["Boltzmann constant"][:2]))
    mu_0 = cast(Quantity, Q_(*physical_constants["mag. constant"][:2]))
    epsilon_0 = cast(
        Quantity, Q_(*physical_constants["vacuum electric permittivity"][:2])
    )
    hbar = cast(Quantity, Q_(*physical_constants["Planck constant over 2 pi"][:2]))
    mu_B = cast(Quantity, Q_(*physical_constants["Bohr magneton"][:2]))
    a_0 = cast(Quantity, Q_(*physical_constants["Bohr radius"][:2]))
    u = cast(Quantity, Q_(*physical_constants["atomic mass constant"][:2]))
    E_h = cast(Quantity, Q_(*physical_constants["Hartree energy"][:2]))
    m_e = cast(Quantity, Q_(*physical_constants["electron mass"][:2]))
    h = cast(Quantity, Q_(*physical_constants["Planck constant"][:2]))
    c = cast(Quantity, Q_(*physical_constants["speed of light in vacuum"][:2]))
    G = cast(Quantity, Q_(*physical_constants["Newtonian constant of gravitation"][:2]))
    e = cast(Quantity, Q_(*physical_constants["elementary charge"][:2]))
    m_p = cast(Quantity, Q_(*physical_constants["proton mass"][:2]))
    m_n = cast(Quantity, Q_(*physical_constants["neutron mass"][:2]))
    alpha = cast(Quantity, Q_(*physical_constants["fine-structure constant"][:2]))
    R_inf = cast(Quantity, Q_(*physical_constants["Rydberg constant"][:2]))
    a_e = cast(Quantity, Q_(*physical_constants["Bohr magneton"][:2]))
