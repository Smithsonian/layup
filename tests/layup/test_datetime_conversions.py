import pytest
from layup.utilities.datetime_conversions import convert_tdb_date_to_julian_date


def test_convert_tdb_date_to_julian_date():
    """Test the conversion of a TDB date string into a Julain date"""
    # Test case 1: Valid TDB date string
    input_tdb_date = "2000-01-01"
    spice_kernel_dir = None  # Use default cache directory
    expected_julian_date = 2451544.500742869  # Example expected value, adjust as needed

    # Call the function
    result = convert_tdb_date_to_julian_date(input_tdb_date, spice_kernel_dir)

    # Check if the result is close to the expected value
    assert abs(result - expected_julian_date) < 1e-6, f"Expected {expected_julian_date}, got {result}"


def test_convert_tdb_date_to_julian_date_raises():
    input_tdb_date = "2000-01-01"
    spice_kernel_dir = "./non_existent_directory"
    with pytest.raises(FileNotFoundError) as exc_info:
        convert_tdb_date_to_julian_date(input_tdb_date, spice_kernel_dir)
        assert exc_info.contains("SPICE kernel file 'naif0012.tls' not found in directory")
