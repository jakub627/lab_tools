import pytest
from lab_tools.rounding import rounded_string


# Test case 1: Basic test without uncertainty
def test_basic_no_uncertainty():
    result = rounded_string(3.14159)
    expected = "x = 3.1"
    assert result == expected


# Test case 2: With uncertainty
def test_with_uncertainty():
    result = rounded_string(3.14159, 0.001)
    expected = "  x = 3.1416\nu_x = 0.0010"
    assert result == expected


# Test case 3: With uncertainty and unit
def test_with_uncertainty_and_unit():
    result = rounded_string(3.14159, 0.001, unit_x="m")
    expected = "  x = 3.1416 [m]\nu_x = 0.0010 [m]"
    assert result == expected


# Test case 4: With scaling for both value and uncertainty
def test_with_scaling():
    result = rounded_string(3141.59, 1, scale_x=3, scale_u_x=3, unit_x="m")
    expected = "  x = 3.1416 [km]\nu_x = 0.0010 [km]"
    assert result == expected


# Test case 5: With name for the measured value and new line
def test_with_name_and_new_line():
    result = rounded_string(1234.56786, 0.12, s_x="Voltage", n=True)
    expected = "\n  Voltage = 1234.57\nu_Voltage =    0.12"
    assert result == expected


# Test case 6: With tabulation and unit scaling
def test_with_tab_and_unit_scaling():
    result = rounded_string(
        0.000567, 0.000001, s_x="Current", unit_x="A", scale_x=-3, scale_u_x=-6, t=True
    )
    expected = "\t  Current = 0.567 [mA]\n\tu_Current = 1 [uA]"
    assert result == expected


# Test case 7: When scale_x equals scale_u_x but no uncertainty
def test_scale_equal_no_uncertainty():
    result = rounded_string(1500, s_x="Length", scale_x=3)
    expected = "Length = 1.5"
    assert result == expected


# Test case 8: Invalid exponent for prefix (should raise ValueError)
def test_invalid_exponent():
    with pytest.raises(ValueError, match="Exponent '100' is not a valid SI exponent."):
        rounded_string(1000, scale_x=100)


# Run the tests
if __name__ == "__main__":
    pytest.main()
