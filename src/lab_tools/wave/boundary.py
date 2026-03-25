from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np


if TYPE_CHECKING:
    from ..typing import ArrayLike, Number, Array


class BoundaryCondition(Protocol):
    fill_value: Array | Number

    def apply(self, arr: ArrayLike) -> Array: ...


class Dirichlet(BoundaryCondition):

    def __init__(
        self,
        fill_value: ArrayLike | Number | None = None,
        axes: ArrayLike | int | None = None,
        edges: Literal["first", "last", "both"] = "both",
    ) -> None:

        self.fill_value: Array | Number = (
            0 if fill_value is None else np.asarray(fill_value)
        )
        self.axes = np.atleast_1d(axes) if axes is not None else None
        if edges not in ("first", "last", "both"):
            raise ValueError("edges must be 'first', 'last' or 'both'")
        self.edges = edges

    def apply(self, arr: ArrayLike) -> Array:
        arr = np.asarray(arr)
        all_axes = tuple(range(arr.ndim))
        if self.axes is not None:
            for ax in self.axes:
                if ax not in all_axes:
                    raise ValueError(
                        f"Axis {ax} is out of bounds for array with ndim={arr.ndim}"
                    )
        axes = self.axes if self.axes is not None else all_axes

        for ax in axes:
            idx: list[slice | int] = [slice(None)] * arr.ndim

            if self.edges in ("first", "both"):
                idx[ax] = 0
                arr[tuple(idx)] = np.broadcast_to(
                    self.fill_value, arr[tuple(idx)].shape
                )

            if self.edges in ("last", "both"):
                idx[ax] = -1
                arr[tuple(idx)] = np.broadcast_to(
                    self.fill_value, arr[tuple(idx)].shape
                )

        return arr


def main():
    arr = np.ones((5, 5))
    a = np.linspace(0, 1, len(arr))
    d = Dirichlet(0)
    print(d.apply(arr))


if __name__ == "__main__":
    main()
