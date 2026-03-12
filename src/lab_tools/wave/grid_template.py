from typing import Any, overload
import numpy as np
from numpy.typing import NDArray


class Grid:
    """
    Multi-dimensional equally or unequally spaced grid with
    position-space and momentum-space (FFT) representations.
    """

    @overload
    def __init__(
        self, lower: float, upper: float, N_points: int, *, dim: int = 2
    ) -> None: ...
    @overload
    def __init__(
        self, lower: list[float], upper: list[float], N_points: list[int]
    ) -> None: ...
    @overload
    def __init__(
        self, lower: float, upper: float, dx: float, *, dim: int = 2
    ) -> None: ...
    @overload
    def __init__(
        self, lower: list[float], upper: list[float], dx: list[float]
    ) -> None: ...

    def __init__(
        self,
        lower: float | list[float],
        upper: float | list[float],
        N_points: int | list[int],
        *,
        dx: float | list[float] | None = None,
        dim: int = 2,
    ) -> None:
        if (
            isinstance(lower, (float, int))
            and isinstance(upper, (float, int))
            and isinstance(N_points, int)
        ):  # dim
            lower = [float(lower)] * dim
            upper = [float(upper)] * dim
            N_points = [int(N_points)] * dim
        elif (
            isinstance(lower, list)
            and isinstance(upper, list)
            and isinstance(N_points, list)
        ):  # list of values
            if not (len(lower) == len(upper) == len(N_points)):
                raise ValueError("lower, upper, N_points must have the same length.")
            lower = list(map(float, lower))
            upper = list(map(float, upper))
            N_points = list(map(int, N_points))
        else:
            raise TypeError("Invalid types for lower, upper, or N_points.")
        if any(n < 2 for n in N_points):
            raise ValueError("Each dimension must have at least 2 points.")
        self.__grid_axes: list[NDArray[np.float64]] = [
            np.linspace(l, u, n, dtype=np.float64)
            for l, u, n in zip(lower, upper, N_points)
        ]

    @property
    def dim(self) -> int:
        return len(self.__grid_axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(ax) for ax in self.__grid_axes)

    @property
    def dx(self) -> tuple[float, ...]:
        return tuple(float(ax[1] - ax[0]) for ax in self.__grid_axes)

    @property
    def lower(self) -> tuple[float, ...]:
        return tuple(float(ax[0]) for ax in self.__grid_axes)

    @property
    def upper(self) -> tuple[float, ...]:
        return tuple(float(ax[-1]) for ax in self.__grid_axes)

    @property
    def equally_spaced(self) -> bool:
        return all(
            np.allclose(np.diff(ax), np.diff(self.__grid_axes[0]))
            for ax in self.__grid_axes
        )

    @property
    def grid_axes(self) -> list[NDArray[np.float64]]:
        return self.__grid_axes

    @property
    def grid(self) -> NDArray[np.float64]:
        """Return coordinate grid as array with shape (dim, N1, N2, ...)."""
        return np.array(np.meshgrid(*self.__grid_axes, indexing="ij"))

    @property
    def grid_k(self) -> NDArray[np.float64]:
        """Return FFT wavenumber grid with shape (dim, N1, N2, ...)."""
        k_axes = [
            2 * np.pi * np.fft.fftfreq(n, d=dx) for n, dx in zip(self.shape, self.dx)
        ]
        return np.array(np.meshgrid(*k_axes, indexing="ij"))

    @property
    def grid_k2(self) -> NDArray[np.float64]:
        """Return squared wavenumber grid with shape (N1, N2, ...)."""
        k_sq = np.sum(self.grid_k**2, axis=0)
        return k_sq

    def get_slice(
        self,
        coord: int,
        slice_dims: int | list[int],
        fixed_indices: int | list[int] | None = None,
    ) -> NDArray[np.float64]:
        """Extract a lower-dimensional slice along specified dimensions."""
        if isinstance(slice_dims, int):
            slice_dims = [slice_dims]
        if not slice_dims:
            raise ValueError("slice_dims must be a non-empty list.")

        all_dims = list(range(self.dim))
        fixed_dims = [d for d in all_dims if d not in slice_dims]

        if isinstance(fixed_indices, int):
            fixed_indices = [fixed_indices] * len(fixed_dims)
        elif fixed_indices is None:
            fixed_indices = [self.shape[d] // 2 for d in fixed_dims]
        elif len(fixed_indices) != len(fixed_dims):
            raise ValueError(
                f"Expected {len(fixed_dims)} fixed indices, got {len(fixed_indices)}"
            )

        indexer: list[slice | int] = [slice(None)] * self.dim
        for d, val in zip(fixed_dims, fixed_indices):
            indexer[d] = val

        return self.grid[coord][tuple(indexer)]

    def get_elements(self) -> dict[str, list[str]]:
        cls = self.__class__
        instance = self

        class_dict = cls.__dict__
        instance_attrs = list(
            key for key in instance.__dict__.keys() if not key.startswith("_")
        )

        properties = [
            k
            for k, v in class_dict.items()
            if isinstance(v, property) and not k.startswith("__")
        ]
        methods = [
            k for k, v in class_dict.items() if callable(v) and not k.startswith("__")
        ]
        static_methods = [
            k for k, v in class_dict.items() if isinstance(v, staticmethod)
        ]
        class_methods = [k for k, v in class_dict.items() if isinstance(v, classmethod)]

        return {
            "attributes": instance_attrs,
            "properties": properties,
            "methods": methods,
            "static_methods": static_methods,
            "class_methods": class_methods,
        }

    def __repr__(self) -> str:
        def fmt_tuple(t: tuple[float, ...]) -> str:
            return "(" + ", ".join(f"{x:.6g}" for x in t) + ")"

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
            [l * other for l in self.lower],
            [u * other for u in self.upper],
            list(self.shape),
        )

    def __truediv__(self, other: float) -> "Grid":
        return self * (1 / other)

    def __add__(self, other: float) -> "Grid":
        return Grid(
            [l + other for l in self.lower],
            [u + other for u in self.upper],
            list(self.shape),
        )

    def __sub__(self, other: float) -> "Grid":
        return self + (-other)

    def __neg__(self) -> "Grid":
        return Grid(
            [-u for u in self.upper], [-l for l in self.lower], list(self.shape)
        )
