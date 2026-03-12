import math
import pytest
from lab_tools.wave.angle import Angle


class TestAngle:

    # =======================
    # Test inicjalizacji
    # =======================
    def test_init_deg(self):
        a = Angle(deg=90)
        assert a.deg == pytest.approx(90)
        assert a.rad == pytest.approx(math.pi / 2)

    def test_init_rad(self):
        a = Angle(rad=math.pi / 2)
        assert a.rad == pytest.approx(math.pi / 2)
        assert a.deg == pytest.approx(90)

    def test_init_error_if_none(self):
        with pytest.raises(ValueError):
            Angle()

    def test_init_error_if_both(self):
        with pytest.raises(ValueError):
            Angle(deg=45, rad=1.0)

    # =======================
    # Test dodawania, odejmowania, mnożenia, dzielenia
    # =======================
    def test_add_sub(self):
        a = Angle(deg=30)
        b = Angle(deg=45)
        c = a + b
        d = b - a
        assert c.deg == pytest.approx(75)
        assert d.deg == pytest.approx(15)

    def test_mul_div(self):
        a = Angle(deg=60)
        b = a * 2
        c = a / 2
        assert b.deg == pytest.approx(120)
        assert c.deg == pytest.approx(30)

    # =======================
    # Test porównania
    # =======================
    def test_eq(self):
        a = Angle(deg=90)
        b = Angle(rad=math.pi / 2)
        assert a == b
        assert a != Angle(deg=91)

    def test_lt_le(self):
        a = Angle(deg=30)
        b = Angle(deg=60)
        assert a < b
        assert a <= b
        assert b > a
        assert b >= a

    # =======================
    # Test normalizacji
    # =======================
    def test_normalized(self):
        a = Angle(deg=370)
        b = a.normalized()
        assert 0 <= b.deg < 360
        assert b.deg == pytest.approx(10)

    def test_normalized_signed(self):
        a = Angle(deg=270)
        b = a.normalized_signed()
        # signed range: [-180, 180)
        assert -180 <= b.deg < 180
        assert b.deg == pytest.approx(-90)

    # =======================
    # Test reprezentacji
    # =======================
    def test_str_repr(self):
        a = Angle(deg=45)
        assert str(a) == "45°"
        assert repr(a) == f"Angle(deg={a.deg:g}, rad={a.rad:g})"

    # =======================
    # Test operacji z obiektami innych typów
    # =======================
    def test_eq_with_non_angle(self):
        a = Angle(deg=45)
        assert a.__eq__(123) is NotImplemented
