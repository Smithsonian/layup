import pytest
from layup.utilities.datetime_conversions import convert_tdb_date_to_julian_date


def test_convert_tdb_date_to_julian_date():
    """Test the conversion of a TDB date string into a Julain date"""
    # Test case 1: Valid TDB date string
    input_tdb_date = "2000-01-01"
    spice_kernel_dir = None  # Use default cache directory
    expected_julian_date = 2451544.500742869  # Example expected value, adjust as needed

    # Call the function
    result, _ = convert_tdb_date_to_julian_date(input_tdb_date, spice_kernel_dir)

    # Check if the result is close to the expected value
    assert abs(result - expected_julian_date) < 1e-6, f"Expected {expected_julian_date}, got {result}"


def test_convert_tdb_date_to_julian_date_raises():
    input_tdb_date = "2000-01-01"
    spice_kernel_dir = "./non_existent_directory"
    with pytest.raises(FileNotFoundError) as exc_info:
        convert_tdb_date_to_julian_date(input_tdb_date, spice_kernel_dir)
        assert exc_info.contains("SPICE kernel file 'naif0012.tls' not found in directory")


def test_convert_tdb_date_to_julian_date_round_trip():
    """Test the round trip conversion of a TDB date string into a Julian date and back."""
    from datetime import datetime, timedelta

    # Define the start and end dates
    start_date = datetime.strptime("1950-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("1990-12-31", "%Y-%m-%d")

    # Generate all dates between the start and end dates
    date_strings = []
    current_date = start_date
    while current_date <= end_date:
        date_string = current_date.strftime("%Y-%m-%d")
        date_strings.append(date_string)

        # Under the hood convert `date_string` to et, and then back to UTC date string.
        _, converted_date = convert_tdb_date_to_julian_date(date_string)
        converted_date_as_string = str(datetime.strptime(converted_date, "%Y-%m-%dT%H:%M:%S.%f").date())

        assert converted_date_as_string == str(
            current_date.date()
        ), f"Round trip conversion failed for date: {date_string}"

        current_date += timedelta(days=90)
