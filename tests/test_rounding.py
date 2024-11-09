import pytest
from lab_tools.rounding import round_to_2

# Test cases for rounding to 2 significant figures or specified uncertainty

# Test 1: Single integer input, no uncertainty
def test_round_to_2_int_no_uncertainty():
    assert round_to_2(1234) == 1200

# Test 2: Single float input, no uncertainty
def test_round_to_2_float_no_uncertainty():
    assert round_to_2(1234.567) == 1200.0

# Test 3: List of integers, no uncertainty
def test_round_to_2_list_int_no_uncertainty():
    assert round_to_2([1234, 5678, 9876]) == [1200, 5700, 9900]

# Test 4: List of floats, no uncertainty
def test_round_to_2_list_float_no_uncertainty():
    assert round_to_2([1234.567, 8765.432, 5432.1]) == [1200.0, 8800.0, 5400.0]

# Test 5: Single integer input with uncertainty
def test_round_to_2_int_with_uncertainty():
    assert round_to_2(1234, u_x=50) == 1234

# Test 6: Single float input with uncertainty
def test_round_to_2_float_with_uncertainty():
    assert round_to_2(1234.567, u_x=50) == 1235.0

# Test 7: List of integers with uncertainty
def test_round_to_2_list_int_with_uncertainty():
    assert round_to_2([1234, 5678, 9876], u_x=[50, 100, 150]) == [1234, 5680, 9880]

# Test 8: List of floats with uncertainty
def test_round_to_2_list_float_with_uncertainty():
    assert round_to_2([1234.567, 8765.432, 5432.1], u_x=[50, 100, 150]) == [
        1235.0,
        8770.0,
        5430.0,
    ]

# Test 9: Single integer input with zero uncertainty
def test_round_to_2_int_zero_uncertainty():
    assert round_to_2(1234, u_x=0) == 1200

# Test 10: Single float input with zero uncertainty
def test_round_to_2_float_zero_uncertainty():
    assert round_to_2(1234.567, u_x=0) == 1200.0

# Test 11: List of integers with zero uncertainty
def test_round_to_2_list_int_zero_uncertainty():
    assert round_to_2([1234, 5678, 9876], u_x=[0, 0, 0]) == [1200, 5700, 9900]

# Test 12: List of floats with zero uncertainty
def test_round_to_2_list_float_zero_uncertainty():
    assert round_to_2([1234.567, 8765.432, 5432.1], u_x=[0, 0, 0]) == [1200.0, 8800.0, 5400.0]

# Test 13: Invalid type for x (string)
def test_round_to_2_invalid_type_x():
    with pytest.raises(TypeError):
        round_to_2("invalid") # type: ignore

# Test 14: Invalid type for u_x (string)
def test_round_to_2_invalid_type_u_x():
    with pytest.raises(TypeError):
        round_to_2(1234, u_x="invalid") # type: ignore

# Test 15: x and u_x lists with different lengths
def test_round_to_2_mismatched_lengths():
    with pytest.raises(ValueError):
        round_to_2([1234, 5678], u_x=[50])

# Test 16: Negative uncertainty
def test_round_to_2_negative_uncertainty():
    with pytest.raises(ValueError):
        round_to_2(1234, u_x=-50)

# Test 17: x is zero
def test_round_to_2_zero_input():
    assert round_to_2(0) == 0
    assert round_to_2([0, 1, 2]) == [0, 1, 2]

# Test 18: x is a very small number (less than 1)
def test_round_to_2_small_number():
    assert round_to_2(0.0001234) == 0.00012
    assert round_to_2([0.0001234, 0.0005678]) == [0.00012, 0.00057]

# Test 19: x is a very large number
def test_round_to_2_large_number():
    assert round_to_2(1234567890) == 1200000000
    assert round_to_2([1234567890, 9876543210]) == [1200000000, 9900000000]
