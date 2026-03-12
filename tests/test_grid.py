from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from lab_tools.wave.grid import Grid


@dataclass
class DefaultInitValues:
    N: int = 101
    dim: int = 2


@dataclass
class InitConfig:
    lower: float | Sequence[float] = -10
    upper: float | Sequence[float] = 10
    dim: int | None = None
    N: int | Sequence[int] | None = None
    dx: float | Sequence[float] | None = None


def grid_from_config(ic: InitConfig):
    return Grid(
        ic.lower,
        ic.upper,
        N=ic.N,
        dim=ic.dim,
        dx=ic.dx,
    )  # type: ignore


class TestGridInit:

    def calc_dx(self, ic: InitConfig) -> tuple[float, ...]:
        dx: tuple[float, ...] = tuple()
        if (
            isinstance(ic.lower, Sequence)
            and isinstance(ic.upper, Sequence)
            and isinstance(ic.N, Sequence)
            and len(ic.lower) == len(ic.upper)
            and len(ic.lower) == len(ic.N)
        ):
            for low, up, n in zip(ic.lower, ic.upper, ic.N):
                dx += ((up - low) / (n - 1),)
        elif (
            isinstance(ic.lower, (int, float))
            and isinstance(ic.upper, (int, float))
            and isinstance(ic.N, int)
        ):
            dx = ((ic.upper - ic.lower) / (ic.N - 1),)
        return dx

    def calc_N(self, ic: InitConfig) -> tuple[int, ...]:
        N: tuple[int, ...] = tuple()
        if (
            isinstance(ic.lower, Sequence)
            and isinstance(ic.upper, Sequence)
            and isinstance(ic.dx, Sequence)
            and len(ic.lower) == len(ic.upper)
            and len(ic.lower) == len(ic.dx)
        ):
            for low, up, d in zip(ic.lower, ic.upper, ic.dx):
                N += (int((up - low) / d + 1),)
        elif (
            isinstance(ic.lower, (int, float))
            and isinstance(ic.upper, (int, float))
            and isinstance(ic.dx, (int, float))
        ):
            N = (int((ic.upper - ic.lower) / ic.dx + 1),)
        return N

    # only lower, upper - number | no dx
    def test_init_0_opt_0(self):

        ic = InitConfig()
        r = grid_from_config(ic)
        assert r.lower == (ic.lower,) * DefaultInitValues.dim
        assert r.upper == (ic.upper,) * DefaultInitValues.dim
        assert r.shape == (DefaultInitValues.N,) * DefaultInitValues.dim
        assert r.dim == DefaultInitValues.dim
        assert all(pytest.approx(rd, d) for rd, d in zip(r.dx, self.calc_dx(ic)))

        # wrong input
        with pytest.raises(ValueError):
            r = Grid("a", 10)  # type: ignore
        with pytest.raises(ValueError):
            r = Grid(10, -10)
        with pytest.raises(ValueError):
            r = Grid([-10, -10], 10)  # type: ignore

    # only lower, upper, dim - numbers| no dx
    def test_init_0_opt_1(self):
        ic = InitConfig(dim=3)
        r = grid_from_config(ic)
        assert ic.dim is not None
        assert r.lower == (ic.lower,) * ic.dim
        assert r.upper == (ic.upper,) * ic.dim
        assert r.shape == (DefaultInitValues.N,) * ic.dim
        assert r.dim == ic.dim
        (pytest.approx(rd, d) for rd, d in zip(r.dx, self.calc_dx(ic)))

        # wrong input
        with pytest.raises(ValueError):
            r = Grid(-10, 10, dim=0)
        with pytest.raises(ValueError):
            r = Grid(-10, 10, dim=-1)
        with pytest.raises(ValueError):
            r = Grid(-10, 10, dim=[1, 2])  # type: ignore
        with pytest.raises(ValueError):
            r = Grid(-10, 10, dim="a")  # type: ignore

    # only lower, upper, N - numbers | no dx
    def test_init_0_opt_2(self):
        ic = InitConfig(N=11)
        r = grid_from_config(ic)

        assert ic.N is not None
        assert r.lower == (ic.lower,) * DefaultInitValues.dim
        assert r.upper == (ic.upper,) * DefaultInitValues.dim
        assert r.shape == (ic.N,) * DefaultInitValues.dim
        assert r.dim == DefaultInitValues.dim
        (pytest.approx(rd, d) for rd, d in zip(r.dx, self.calc_dx(ic)))

        # wrong input
        with pytest.raises(ValueError):
            r = Grid(-10, 10, N=0)
        with pytest.raises(ValueError):
            r = Grid(-10, 10, N=-1)
        with pytest.raises(ValueError):
            r = Grid(-10, 10, N=[1, 2])  # type: ignore
        with pytest.raises(ValueError):
            r = Grid(-10, 10, N="a")  # type: ignore

    # only lower, upper, N, dim - number | no dx
    def test_init_0_opt_3(self):
        ic = InitConfig(N=11, dim=3)
        r = grid_from_config(ic)

        assert ic.N is not None
        assert ic.dim is not None
        assert r.lower == (ic.lower,) * ic.dim
        assert r.upper == (ic.upper,) * ic.dim
        assert r.shape == (ic.N,) * ic.dim
        assert r.dim == ic.dim
        (pytest.approx(rd, d) for rd, d in zip(r.dx, self.calc_dx(ic)))

    # only lower, upper - sequence | no dx
    def test_init_1_opt_0(self):

        ic = InitConfig((-10, -5), (10, 5))
        r = grid_from_config(ic)
        assert r.lower == ic.lower
        assert r.upper == ic.upper
        assert r.shape == (DefaultInitValues.N,) * DefaultInitValues.dim
        assert r.dim == DefaultInitValues.dim
        assert all(pytest.approx(rd, d) for rd, d in zip(r.dx, self.calc_dx(ic)))

        # wrong input
        with pytest.raises(ValueError):
            r = Grid((-10, -10), (10, 10, 10))
        with pytest.raises(ValueError):
            r = Grid((-10, -10), (10, -10))
        with pytest.raises(ValueError):
            r = Grid((-10, -10), (10, ""))  # type: ignore

    # only lower, upper, N - sequence | no dx
    def test_init_1_opt_1(self):

        ic = InitConfig((-10, -5, -1), (10, 5, 1), N=(5, 11, 5))
        r = grid_from_config(ic)
        assert isinstance(ic.N, Sequence)
        assert r.lower == ic.lower
        assert r.upper == ic.upper
        assert r.shape == ic.N
        assert r.dim == len(ic.N)
        (pytest.approx(rd, d) for rd, d in zip(r.dx, self.calc_dx(ic)))

        # wrong input
        with pytest.raises(ValueError):
            r = Grid((-10, -10), (10, 10), N=1)  # type: ignore
        with pytest.raises(ValueError):
            r = Grid((-10, -10), (10, 10), N=(10, 10, 10))
        with pytest.raises(ValueError):
            r = Grid((-10, -10), (10, 10), N=(10, ""))  # type: ignore
        with pytest.raises(ValueError):
            r = Grid((-10, -10), (10, 10), N=(0, 0))

    # only lower, upper, dx - number | no N
    def test_init_2_opt_0(self):

        ic = InitConfig(dx=1)
        r = grid_from_config(ic)
        assert r.lower == (ic.lower,) * DefaultInitValues.dim
        assert r.upper == (ic.upper,) * DefaultInitValues.dim
        assert r.dx == (ic.dx,) * DefaultInitValues.dim
        assert r.dim == DefaultInitValues.dim
        (pytest.approx(rn, n) for rn, n in zip(r.shape, self.calc_N(ic)))

        # wrong input
        with pytest.raises(ValueError):
            r = Grid(-10, 10, dx=0)  # type: ignore
        with pytest.raises(ValueError):
            r = Grid(-10, 10, dx=-1)  # type: ignore
        with pytest.raises(ValueError):
            r = Grid(-10, 10, dx="")  # type: ignore
        with pytest.raises(ValueError):
            r = Grid(-10, 10, dx=[1, 1])  # type: ignore

    # only lower, upper, dx, dim - number | no N
    def test_init_2_opt_1(self):

        ic = InitConfig(dx=1, dim=3)
        r = grid_from_config(ic)
        assert ic.dim is not None
        assert r.lower == (ic.lower,) * ic.dim
        assert r.upper == (ic.upper,) * ic.dim
        assert r.dx == (ic.dx,) * ic.dim
        assert r.dim == ic.dim
        (pytest.approx(rn, n) for rn, n in zip(r.shape, self.calc_N(ic)))

        # wrong input
        with pytest.raises(ValueError):
            r = Grid(-10, 10, dx=1, dim=0)  # type: ignore
        with pytest.raises(ValueError):
            r = Grid(-10, 10, dx=1, dim=-1)  # type: ignore
        with pytest.raises(ValueError):
            r = Grid(-10, 10, dx=1, dim="")  # type: ignore
        with pytest.raises(ValueError):
            r = Grid(-10, 10, dx=1, dim=[1, 2])  # type: ignore

    # only lower, upper, dx - sequence | no N
    def test_init_2_opt_2(self):

        ic = InitConfig(lower=(-10, -5, -1), upper=(10, 5, 1), dx=(1, 2, 3))
        r = grid_from_config(ic)
        assert r.lower == ic.lower
        assert r.upper == ic.upper
        assert r.dx == ic.dx
        assert r.dim == 3
        (pytest.approx(rn, n) for rn, n in zip(r.shape, self.calc_N(ic)))

        # wrong input
        with pytest.raises(ValueError):
            r = Grid((-10, -10, -10), (10, 10, 10), dx=1)  # type: ignore
        with pytest.raises(ValueError):
            r = Grid((-10, -10, -10), (10, 10, 10), dx=(1, 1))  # type: ignore
        with pytest.raises(ValueError):
            r = Grid((-10, -10, -10), (10, 10, 10), dx=(1, 1, "a"))  # type: ignore
        with pytest.raises(ValueError):
            r = Grid((-10, -10, -10), (10, 10, 10), dx=(1, 1, 0))  # type: ignore


class TestGridProperties:
    def test_dim(self):
        r = Grid(-10, 10, dim=3)
        assert r.dim == 3
        r = Grid(-10, 10, dim=100)
        assert r.dim == 100
        r = Grid(-10, 10)
        assert r.dim == DefaultInitValues.dim

    def test_axes(self):
        r = Grid((-10, -5), (10, 5), N=(11, 7))
        assert list(r.axes[0]) == np.linspace(-10, 10, 11).tolist()
        assert list(r.axes[1]) == np.linspace(-5, 5, 7).tolist()

    def test_grid(self):
        r = Grid((-10, -5, -15), (10, 5, 10), N=(11, 7, 6))
        x = np.linspace(-10, 10, 11)
        y = np.linspace(-5, 5, 7)
        z = np.linspace(-15, 10, 6)
        assert not (r.grid - np.array(np.meshgrid(x, y, z, indexing="ij"))).any()

    def test_shape(self):
        r = Grid(-10, 10)
        assert r.shape == (DefaultInitValues.N,) * DefaultInitValues.dim
        r = Grid(-10, 10, N=100)
        assert r.shape == (100,) * DefaultInitValues.dim
        r = Grid(-10, 10, N=2)
        assert r.shape == (2,) * DefaultInitValues.dim
        r = Grid(-10, 10, N=2, dim=5)
        assert r.shape == (2,) * 5
        r = Grid(-1, 1, N=123)
        assert r.shape == (123, 123)
        r = Grid(-1, 1, N=3, dim=4)
        assert r.shape == (3,) * 4
        r = Grid(
            (-1,) * 4,
            (1,) * 4,
            N=(2, 3, 4, 5),
        )
        assert r.shape == (2, 3, 4, 5)

    def test_lower_upper(self):
        r = Grid(-10, 10)
        assert r.lower == (-10,) * DefaultInitValues.dim
        assert r.upper == (10,) * DefaultInitValues.dim
        r = Grid(-10, 10, dim=4)
        assert r.lower == (-10,) * 4
        assert r.upper == (10,) * 4
        r = Grid((1, 2, 3, 4, 5), (2, 3, 4, 5, 6))
        assert r.lower == (1, 2, 3, 4, 5)
        assert r.upper == (2, 3, 4, 5, 6)

    def test_equally_spaced(self):
        r = Grid(-1, 1, dim=3)
        assert r.equally_spaced
        r = Grid((0, 0, 0), (1, 1, 1), N=(3, 3, 3))
        assert r.equally_spaced
        r = Grid((0, 0, 0), (1, 1, 1), N=(2, 3, 3))
        assert not r.equally_spaced


class TestGridOperations:
    def test_eq(self):
        r1 = Grid(2, 8, dim=4, N=9)
        r2 = Grid((2,) * 4, (8,) * 4, N=(9,) * 4)
        assert (
            r1 == r2
            and r1.dim == r2.dim
            and r1.shape == r2.shape
            and r1.dx == r2.dx
            and r1.lower == r2.lower
            and r1.upper == r2.upper
        )
        r2 = Grid((2,) * 3 + (1,), (8,) * 3 + (2,), N=(9,) * 4)
        assert r1 != r2

    def test_mul(self):
        assert Grid(-1, 1) * 2 == Grid(-2, 2)
        assert 4.5 * Grid(0, 1) == Grid(0, 4.5)

    def test_add(self):
        assert Grid(-1, 1) + 3.4 == Grid(-1 + 3.4, 1 + 3.4)
        assert 1 / 3 + Grid(-1, 1) == Grid(1 / 3 - 1, 1 / 3 + 1)

        assert -Grid(-3, 10) == Grid(-10, 3)

        assert 1 / 7 - Grid(-3, 10) == Grid(-10 + 1 / 7, 3 + 1 / 7)
        assert Grid(-3, 10) - 5 / 8 == Grid(-3 - 5 / 8, 10 - 5 / 8)
