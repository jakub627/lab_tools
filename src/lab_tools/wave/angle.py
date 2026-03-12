from typing import overload
import math


class Angle:

    @overload
    def __init__(self, *, deg: float) -> None: ...

    @overload
    def __init__(self, *, rad: float) -> None: ...

    def __init__(self, *, deg: float | None = None, rad: float | None = None) -> None:
        value = math.radians(deg) if deg is not None else rad
        if value is None:
            raise ValueError("Provide exactly one of: deg or rad")
        if deg is not None and rad is not None:
            raise ValueError("Provide exactly one of: deg or rad")
        self._rad = value
        self._deg = math.degrees(self._rad)

    @property
    def deg(self) -> float:
        return self._deg

    @property
    def rad(self) -> float:
        return self._rad

    def normalized(self) -> "Angle":
        return Angle(rad=self._rad % (2 * math.pi))

    def normalized_signed(self) -> "Angle":
        x = (self._rad + math.pi) % (2 * math.pi) - math.pi
        return Angle(rad=x)

    def __add__(self, other: "Angle") -> "Angle":
        return Angle(rad=self.rad + other.rad)

    def __sub__(self, other: "Angle") -> "Angle":
        return Angle(rad=self.rad - other.rad)

    def __mul__(self, other: float) -> "Angle":
        return Angle(rad=self.rad * other)

    def __truediv__(self, other: float) -> "Angle":
        return Angle(rad=self.rad / other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Angle):
            return NotImplemented
        return math.isclose(self.rad, other.rad, rel_tol=1e-9)

    def __lt__(self, other: "Angle") -> bool:
        return self.rad < other.rad

    def __le__(self, other: "Angle") -> bool:
        return self.rad <= other.rad

    def __repr__(self) -> str:
        return f"Angle(deg={self.deg:g}, rad={self.rad:g})"

    def __str__(self) -> str:
        return f"{self.deg:g}°"


def main():
    x = Angle(deg=90)
    print(repr(x))
    print(x + Angle(deg=10))


if __name__ == "__main__":
    main()
