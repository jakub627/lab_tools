import math

from matplotlib import pyplot as plt
from numpy import complex128, dtype, float64, int64, ndarray
import numpy as np
from lab_tools.wave.grid import Grid
from typing import (
    Literal,
    Sequence,
    TypeVar,
    Generic,
    overload,
)

T = TypeVar("T", float64, complex128)


class WaveFunction(Generic[T]):
    def __init__(
        self,
        psi: ndarray[tuple[int, ...], dtype[T]],
        grid: Grid,
    ) -> None:

        if grid.shape != psi.shape:
            raise ValueError(
                f"Shape mismatch: gird has shape of {grid.shape}, but arr has shape of {psi.shape}"
            )
        self._psi: ndarray[tuple[int, ...], dtype[T]] = psi
        self._grid = grid
        raise NotImplementedError

    @property
    def psi(self) -> ndarray[tuple[int, ...], dtype[T]]:
        return self._psi

    @property
    def density(self) -> ndarray[tuple[int, ...], dtype[float64]]:
        return np.abs(self._psi) ** 2

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def norm(self) -> float:
        density = self.density
        vol_elements = np.prod(self.grid.dx, dtype=float64)
        integral = np.sum(density) * vol_elements
        norm = math.sqrt(integral)
        return norm

    @overload
    def normalize(self, inplace: Literal[True]) -> None: ...
    @overload
    def normalize(self, inplace: Literal[False] = False) -> "WaveFunction[T]": ...
    def normalize(self, inplace: bool = False) -> "WaveFunction[T] | None":
        norm = self.norm
        if norm == 0:
            raise ValueError("Cannot normalize a wave function with zero norm")
        normalized_psi = self._psi / norm
        if inplace:
            self._psi = normalized_psi.astype(self._psi.dtype)
        else:
            return WaveFunction(normalized_psi.astype(self._psi.dtype), self._grid)

    def dot(self, other: "WaveFunction"):
        if self._grid != other._grid:
            raise ValueError("Incompatible grids")

        vol = np.prod(self.grid.dx, dtype=np.float64)
        return float(np.sum(np.conjugate(self._psi) * other._psi) * vol)

    def conjugate(self) -> "WaveFunction[T] | WaveFunction[complex128]":
        return WaveFunction(np.conjugate(self._psi), self._grid)

    def laplacian(self) -> "WaveFunction[T]":
        psi = self._psi
        result = np.zeros_like(psi)

        for axis, dx in enumerate(self.grid.dx):
            result += (
                np.roll(psi, 1, axis=axis) - 2 * psi + np.roll(psi, -1, axis=axis)
            ) / dx**2

        return WaveFunction(result.astype(self._psi.dtype), self._grid)

    def copy(self) -> "WaveFunction[T]":
        return WaveFunction(self._psi.copy(), self._grid)

    @property
    def phase(self) -> ndarray[tuple[int, ...], dtype[float64]]:
        return np.angle(self._psi).astype(float64)

    @property
    def amplitude(self) -> ndarray[tuple[int, ...], dtype[float64]]:
        return np.abs(self._psi)

    def astype(self, dtype: type) -> "WaveFunction":
        return WaveFunction(self._psi.astype(dtype), self._grid)

    def __abs__(self) -> float:
        return self.norm

    def __mul__(
        self, other: float | complex
    ) -> "WaveFunction[complex128] | WaveFunction[T]":
        if isinstance(other, complex) or np.iscomplexobj(self._psi):
            result_psi = (self._psi * other).astype(np.complex128)
            return WaveFunction(result_psi, self._grid)
        else:
            result_psi = (self._psi * other).astype(self._psi.dtype)
            return WaveFunction(result_psi, self._grid)

    __rmul__ = __mul__

    def __truediv__(
        self, other: float | complex
    ) -> "WaveFunction[complex128] | WaveFunction[T]":
        if other == 0:
            raise ZeroDivisionError("division by zero")
        return self.__mul__(1 / other)

    def __rtruediv__(
        self, other: float | complex
    ) -> "WaveFunction[complex128] | WaveFunction[T]":
        if np.any(self._psi == 0):
            raise ZeroDivisionError("division by zero in WaveFunction")
        if isinstance(other, complex) or np.iscomplexobj(self._psi):
            result_psi = (other / self._psi).astype(np.complex128)
            return WaveFunction(result_psi, self._grid)
        else:
            result_psi = (other / self._psi).astype(self._psi.dtype)
            return WaveFunction(result_psi, self._grid)

    def __add__(
        self, other: "float | complex | WaveFunction"
    ) -> "WaveFunction[complex128]|WaveFunction[T]":
        if isinstance(other, WaveFunction):
            if not self._grid == other._grid:
                raise ValueError("Incompatible grids")
            if np.iscomplexobj(self._psi) or np.iscomplexobj(other._psi):
                dt = np.complex128
            else:
                dt = self._psi.dtype
            result_psi = (self._psi + other._psi).astype(dt)
        else:  # float, complex
            if isinstance(other, complex) or np.iscomplexobj(self._psi):
                dt = np.complex128
            else:
                dt = self._psi.dtype
            result_psi = (self._psi + other).astype(dt)
        return WaveFunction(result_psi, self._grid)

    __radd__ = __add__

    def __neg__(self):
        return self.__mul__(-1)

    def __sub__(self, other: "float | complex | WaveFunction"):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __repr__(self) -> str:
        return (
            f"WaveFunction("
            f"shape={self._psi.shape}, "
            f"dtype={self._psi.dtype}, "
            f"norm={self.norm:g})"
        )

    __str__ = __repr__

    def __array__(self, dtype=None):
        if dtype is not None:
            return np.asarray(self._psi, dtype=dtype)
        return np.asarray(self._psi)

    @property
    def real(self) -> ndarray[tuple[int, ...], dtype[float64]]:
        return self._psi.real

    @property
    def imag(self) -> ndarray[tuple[int, ...], dtype[float64]]:
        return self._psi.imag

    def get_slice(
        self, axis: Sequence[int] | int, offset: Sequence[int] | int | None = None
    ) -> ndarray[tuple[int, ...], dtype[float64]]:

        ax = tuple(np.atleast_1d(axis))
        arr = self.grid.grid
        shape = arr.shape

        spatial_shape = shape[1:]
        nd = len(spatial_shape)

        # default offsets = center
        offset_full = [n // 2 for n in spatial_shape]

        # overwrite offsets for reduced axes
        if offset is not None:
            if isinstance(offset, int):
                offset = tuple(offset for _ in range(nd - len(ax)))
            j = 0
            for i in range(nd):
                if i not in ax:
                    offset_full[i] = offset[j]
                    j += 1

        # build slice
        slc = [slice(None)]  # dim

        for i in range(nd):
            if i in ax:
                slc.append(slice(None))
            else:
                o = offset_full[i]
                slc.append(slice(o, o + 1))

        result = arr[tuple(slc)][ax,]
        return result.squeeze()


@overload
def wave_function(
    arr: ndarray[tuple[int, ...], dtype[float64]], grid: Grid
) -> WaveFunction[float64]: ...
@overload
def wave_function(
    arr: ndarray[tuple[int, ...], dtype[complex128]], grid: Grid
) -> WaveFunction[complex128]: ...
def wave_function(
    arr: ndarray[tuple[int, ...], dtype[float64 | complex128]], grid: Grid
) -> WaveFunction[complex128] | WaveFunction[float64]:
    raise NotImplementedError
    dt: dtype[float64 | complex128 | int64] = arr.dtype
    if dt in (float64, complex128, int64):
        if dt == int64:
            arr = arr.astype(float64)
        return WaveFunction[dt](arr, grid)
    else:
        raise ValueError(f"Expected types: `float64` or `complex128`, but got `{dt}`")


def main():
    N = (101, 101, 101)
    dim = len(N)

    grid = Grid((-1,) * dim, (1,) * dim, N=N)
    x = grid.grid[0]
    y = grid.grid[1]
    # z = grid.grid[2]

    arr: ndarray = x**2 + y**2  # + z**2
    psi = wave_function(arr, grid)
    psi.normalize(inplace=True)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.pcolormesh(psi.get_slice(0), psi.get_slice(1), psi.psi[:, :, 0])
    plt.show()


if __name__ == "__main__":
    main()
