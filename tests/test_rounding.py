import pytest
from lab_tools.rounding import round_to_2

# Test cases for rounding to 2 significant figures or specified uncertainty


# Single integer input, no uncertainty
def test_round_to_2_int_no_uncertainty():
    assert round_to_2(1234) == 1200


# Single float input, no uncertainty
def test_round_to_2_float_no_uncertainty():
    assert round_to_2(1234.567) == 1200.0


# Single float input, no uncertainty
def test_round_to_2_small_float_no_uncertainty():
    assert round_to_2(0.006567) == 0.0066


# List of integers, no uncertainty
def test_round_to_2_list_int_no_uncertainty():
    assert round_to_2([1234, 5678, 9876]) == [1200, 5700, 9900]


# List of floats, no uncertainty
def test_round_to_2_list_float_no_uncertainty():
    assert round_to_2([1234.567, 0.007765, 0.6593]) == [1200.0, 0.0078, 0.66]


# Single integer input with uncertainty
def test_round_to_2_int_with_uncertainty():
    assert round_to_2(1234, u_x=50) == 1234


# Single integer input with uncertainty
def test_round_to_2_int_with_uncertainty2():
    assert round_to_2(876457, u_x=6789) == 876500


# Single float input with uncertainty
def test_round_to_2_float_with_uncertainty():
    assert round_to_2(87645787.567, u_x=9889) == 87645800.0


# Single float input with uncertainty
def test_round_to_2_float_with_uncertainty2():
    assert round_to_2(87645787.00868567, u_x=0.00008567) == 87645787.008686


# List of integers with uncertainty
def test_round_to_2_list_int_with_uncertainty():
    assert round_to_2([98765765, 76556789, 98765678], u_x=[98765, 98778, 98708765]) == [
        98766000,
        76557000,
        99000000,
    ]


# List of floats with uncertainty
def test_round_to_2_list_float_with_uncertainty():
    assert round_to_2(
        [98765765.098765678, 76556789.0086698765, 98765678.997556],
        u_x=[0.0888876, 0.000086, 0.008766],
    ) == [
        98765765.099,
        76556789.00867,
        98765678.9976,
    ]


# Single integer input with zero uncertainty
def test_round_to_2_int_zero_uncertainty():
    assert round_to_2(9876554, u_x=0) == 9900000


# Single float input with zero uncertainty
def test_round_to_2_float_zero_uncertainty():
    assert round_to_2(85644.9076, u_x=0) == 86000.0


# List of integers with zero uncertainty
def test_round_to_2_list_int_zero_uncertainty():
    assert round_to_2([1234, 5678, 9876], u_x=[0, 0, 0]) == [1200, 5700, 9900]


# List of floats with zero uncertainty
def test_round_to_2_list_float_zero_uncertainty():
    assert round_to_2([1234.567, 8765.432, 5432.1], u_x=[0, 0, 0]) == [
        1200.0,
        8800.0,
        5400.0,
    ]


# Invalid type for x (string)
def test_round_to_2_invalid_type_x():
    with pytest.raises(TypeError):
        round_to_2("invalid")  # type: ignore


# Invalid type for u_x (string)
def test_round_to_2_invalid_type_u_x():
    with pytest.raises(TypeError):
        round_to_2(1234, u_x="invalid")  # type: ignore


# x and u_x lists with different lengths
def test_round_to_2_mismatched_lengths():
    with pytest.raises(ValueError):
        round_to_2([1234, 5678], u_x=[50])


# Negative uncertainty
def test_round_to_2_negative_uncertainty():
    with pytest.raises(ValueError):
        round_to_2(1234, u_x=-50)


# x is zero
def test_round_to_2_zero_input():
    assert round_to_2(0) == 0
    assert round_to_2([0, 1, 2]) == [0, 1, 2]


# x is a very small number (less than 1)
def test_round_to_2_small_number():
    assert round_to_2(0.0001234) == 0.00012
    assert round_to_2([0.0001234, 0.0005678]) == [0.00012, 0.00057]


# x is a very large number
def test_round_to_2_large_number():
    assert round_to_2(1234567890) == 1200000000
    assert round_to_2([1234567890, 9876543210]) == [1200000000, 9900000000]


# there is zero in one of the lists
def test_round_to_2_zero_in_list():
    assert round_to_2([1234567890, 0, 0.0001234, 0.0]) == [1200000000, 0, 0.00012, 0.0]
    assert round_to_2([1234567890, 0, 0.0001234, 0.0], [0, 3213, 0, 0.00653]) == [
        1200000000,
        0,
        0.00012,
        0.0,
    ]
