from dataclasses import dataclass
from collections.abc import Sequence
from typing import overload
import numpy as np
from numpy import ndarray, float64, dtype


class Grid:

    @dataclass
    class DefaultInitValues:
        N: int = 101
        dim: int = 2

    @overload
    def __init__(
        self,
        lower: float,
        upper: float,
        *,
        N: int | None = None,
        dim: int | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
        *,
        N: Sequence[int] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        lower: float,
        upper: float,
        *,
        dx: float,
        dim: int | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
        *,
        dx: Sequence[float],
    ) -> None: ...

    def _get_initial_values(
        self,
        lower: float | Sequence[float],
        upper: float | Sequence[float],
        *,
        dx: float | Sequence[float] | None = None,
        dim: int | None = None,
        N: int | Sequence[int] | None = None,
    ):
        if dim is not None and not isinstance(dim, int):
            raise ValueError("Dimensions must be int")

        if dx is None:
            if (  # lower, upper, N, dim - number
                isinstance(lower, (float, int))
                and isinstance(upper, (float, int))
                and (isinstance(N, int) or N is None)
            ):
                dim = dim if dim is not None else self.DefaultInitValues.dim
                N = N if N is not None else self.DefaultInitValues.N
                _lower: tuple[float, ...] = (float(lower),) * dim
                _upper: tuple[float, ...] = (float(upper),) * dim
                _N: tuple[int, ...] = (N,) * dim
                _dim: int = dim

            elif (  # lower, upper, N - sequence
                isinstance(lower, Sequence)
                and isinstance(upper, Sequence)
                and (isinstance(N, Sequence) or N is None)
                and dim is None
            ):
                N = N if N is not None else (self.DefaultInitValues.N,) * len(lower)
                if not all(
                    isinstance(v, (int, float)) for seq in (lower, upper) for v in seq
                ):
                    raise ValueError("All values must be float")
                if not all(isinstance(v, (int, float)) for seq in (N,) for v in seq):
                    raise ValueError("All values must be int")

                if not (len(upper) == len(lower) and len(lower) == len(N)):
                    raise ValueError("lower, upper and N must have the same length.")
                _lower = tuple(map(float, lower))
                _upper = tuple(map(float, upper))
                _N = tuple(map(int, N))
                _dim = len(_N)
            else:
                raise ValueError(
                    "lower, upper and must be either float or sequence of floats and dim must be int."
                )

            _dx = tuple((up - low) / (n - 1) for up, low, n in zip(_upper, _lower, _N))
        else:
            if not all(isinstance(d, (int, float)) for d in np.atleast_1d(dx).tolist()):
                raise ValueError("dx must be float or sequence of float.")
            if not all(d > 0 for d in np.atleast_1d(dx)):
                raise ValueError("dx must be greater than zero.")
            if N is not None:
                raise NotImplementedError
            if (  # lower, upper, dx, dim - number
                isinstance(lower, (float, int))
                and isinstance(upper, (float, int))
                and isinstance(dx, (float, int))
                and (isinstance(dim, int) or dim is None)
            ):
                dim = dim if dim is not None else self.DefaultInitValues.dim
                _lower: tuple[float, ...] = (float(lower),) * dim
                _upper: tuple[float, ...] = (float(upper),) * dim

                _dim = dim
                _dx: tuple[float, ...] = (dx,) * dim
            elif (  # lower, upper, dx - sequence
                isinstance(lower, Sequence)
                and isinstance(upper, Sequence)
                and isinstance(dx, Sequence)
                and dim is None
            ):
                N = N if N is not None else (self.DefaultInitValues.N,) * len(lower)

                if not (len(upper) == len(lower) and len(lower) == len(dx)):
                    raise ValueError("lower, upper and dx must have the same length.")
                _lower = tuple(map(float, lower))
                _upper = tuple(map(float, upper))
                _dim = len(_lower)
                _dx = tuple(map(float, dx))
            else:
                raise ValueError("dx must be float or sequence of float.")
            _N = tuple(
                int((up - low) / d + 1) for low, up, d in zip(_lower, _upper, _dx)
            )

        if not all(low < high for low, high in zip(_lower, _upper)):
            raise ValueError("Lower boundaries must be lower than upper.")
        if not all(n > 0 for n in _N):
            raise ValueError("Number of points must be grater than zero.")
        if not all(d > 0 for d in _dx):
            raise ValueError("dx must be grater than zero.")
        if _dim < 1:
            raise ValueError("Grid dimensions must be grater thn zero.")

        @dataclass
        class ParamsDict:
            lower: tuple[float, ...]
            upper: tuple[float, ...]
            dx: tuple[float, ...]
            N: tuple[int, ...]
            dim: int

        return ParamsDict(
            lower=_lower,
            upper=_upper,
            dx=_dx,
            N=_N,
            dim=_dim,
        )

    def __init__(
        self,
        lower: float | Sequence[float],
        upper: float | Sequence[float],
        *,
        dx: float | Sequence[float] | None = None,
        dim: int | None = None,
        N: int | Sequence[int] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        `lower` : `float | Sequence[float]`
            Lower limits of grid, if not a number, then `upper`, `N`, and `dx` must be matching lengths
        `upper` : `float | Sequence[float]`
            Upper limits of grid, if not a number, then `lower`, `N`, and `dx` must be matching lengths
        `dx` : `float | Sequence[float] | None`, optional
            Grid step in each axis, by default None. If specified, then `N` is calculated automatically to fit whole number od `dx` inside of `(lower, upper)` range. Upper boundary is then set to: `upper = lower + (N - 1) * dx`
        `dim` : `int | None`, optional
            Grid dimensions, by default 2
        `N` : `int | Sequence[int] | None`, optional
            Number of points in ech axis, by default 101


        """
        params = self._get_initial_values(
            lower=lower,
            upper=upper,
            dx=dx,
            dim=dim,
            N=N,
        )
        self._lower: tuple[float, ...] = params.lower
        self._upper: tuple[float, ...] = params.upper
        self._dim: int = params.dim
        self._N: tuple[int, ...] = params.N
        self._dx: tuple[float, ...] = params.dx
        self._axes: tuple[ndarray[tuple[int, ...], dtype[float64]], ...] = tuple(
            np.linspace(low, high, n, dtype=float64)
            for low, high, n in zip(self._lower, self._upper, self._N)
        )

    @property
    def dim(self) -> int:
        """Dimensions of the grid."""
        return self._dim

    @property
    def x(self) -> ndarray[tuple[int, ...], dtype[float64]]:
        if self._dim < 1:
            raise AttributeError("Grid has no x axis")
        return self._axes[0]

    @property
    def y(self) -> ndarray[tuple[int, ...], dtype[float64]]:
        if self._dim < 2:
            raise AttributeError("Grid has no y axis")
        return self._axes[1]

    @property
    def z(self) -> ndarray[tuple[int, ...], dtype[float64]]:
        if self._dim < 3:
            raise AttributeError("Grid has no z axis")
        return self._axes[2]

    @property
    def axes(self) -> tuple[ndarray[tuple[int, ...], dtype[float64]], ...]:
        """Array of coordinates in each axis, shape=(dim,)."""
        return self._axes

    @property
    def grid(self) -> ndarray[tuple[int, ...], dtype[float64]]:
        """Coordinates grid as array, `shape=(dim, N1, N2, ...)`."""
        return np.array(np.meshgrid(*self._axes, indexing="ij"))

    @property
    def shape(self) -> tuple[int, ...]:
        """Tuple of number of elements in each axis, `shape=(dim,)`."""
        return self._N

    @property
    def lower(self) -> tuple[float, ...]:
        """Lower limits of the grid, `shape=(dim,)`."""
        return self._lower

    @property
    def upper(self) -> tuple[float, ...]:
        """Upper limits of the grid, `shape=(dim,)`."""
        return self._upper

    @property
    def dx(self) -> tuple[float, ...]:
        """Grid step in each axis, `shape=(dim,)`."""
        return self._dx

    @property
    def equally_spaced(self) -> bool:
        """Returns `True` if `dx` is equal across all axes, `False` otherwise."""
        return not any(np.diff(self._dx))

    @property
    def k_grid(self) -> ndarray[tuple[int, ...], dtype[float64]]:
        """FFT wavenumber grid, `shape=(dim, N1, N2, ...)`."""
        k_axes = [
            2 * np.pi * np.fft.fftfreq(n, d=dx) for n, dx in zip(self.shape, self.dx)
        ]
        return np.array(np.meshgrid(*k_axes, indexing="ij"), dtype=float64)

    @property
    def k2_grid(self):
        """Squared FFT wavenumber grid, `shape=(N1, N2, ...)`."""
        k_sq = np.sum(self.k_grid**2, axis=0)
        return k_sq

    @property
    def limits(self):
        """Limits of each axis of the grid, `shape=(dim, 2)`."""
        return np.column_stack((self.lower, self.upper))

    def __repr__(self) -> str:
        def fmt_tuple(t: tuple[float, ...]) -> str:
            return "(" + ", ".join(f"{x:g}" for x in t) + ")"

        return (
            "Grid("
            f"lower={fmt_tuple(self.lower)}, "
            f"upper={fmt_tuple(self.upper)}, "
            f"shape={self.shape}, "
            f"dx={fmt_tuple(self.dx)}, "
            f"dim={self.dim}"
            ")"
        )

    __str__ = __repr__

    def __mul__(self, other: float) -> "Grid":
        return Grid(
            [low * other for low in self.lower],
            [high * other for high in self.upper],
            N=self.shape,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: float) -> "Grid":
        return self * (1 / other)

    def __add__(self, other: float) -> "Grid":
        return Grid(
            [low + other for low in self.lower],
            [high + other for high in self.upper],
            N=self.shape,
        )

    __radd__ = __add__

    def __sub__(self, other: float) -> "Grid":
        return self + (-other)

    def __rsub__(self, other: float) -> "Grid":
        return -self + other

    def __neg__(self) -> "Grid":
        return Grid(
            [-high for high in self.upper],
            [-low for low in self.lower],
            N=self.shape,
        )

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Grid):
            if not self.dim == value.dim:
                return False
            if not self.shape == value.shape:
                return False
            if not self.dx == value.dx:
                return False
            if not self.lower == value.lower:
                return False
            if not self.upper == value.upper:
                return False

            return True
        else:
            return False


def main():
    pass


if __name__ == "__main__":
    main()
