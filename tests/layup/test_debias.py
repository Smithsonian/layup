import pytest
import pooch
import numpy as np


from layup.utilities.debiasing import debias, generate_bias_dict, MPC_CATALOGS


def test_generate_bias_dict():
    """Test the generate_bias_dict function to ensure it returns a dictionary with the correct structure."""

    cache_dir = pooch.os_cache("layup")
    # Call the function to generate the bias dictionary
    bias_dict = generate_bias_dict(cache_dir=cache_dir)

    # Check that the returned object is a dictionary
    assert isinstance(bias_dict, dict), "The returned object should be a dictionary."

    # Check that the dictionary contains expected keys
    bias_keys = MPC_CATALOGS.values()
    for key in bias_keys:
        assert key in bias_dict, f"The key '{key}' is missing from the bias dictionary."

    # Check that each key has a list as its value
    for key in bias_keys:
        assert isinstance(bias_dict[key], dict), f"The value for key '{key}' should be a list."

    # Optionally, check that lists are not empty (if applicable)
    for key in bias_keys:
        assert len(bias_dict[key].keys()) == 4, f"The list for key '{key}' should not be empty."
        columns = ["ra", "dec", "pm_ra", "pm_dec"]
        assert all(
            col in bias_dict[key] for col in columns
        ), f"Missing expected columns in bias dictionary for key '{key}'."
        assert all(
            isinstance(bias_dict[key][col], np.ndarray) for col in columns
        ), f"Values for key '{key}' should be lists."


def test_generate_bias_dict_file_not_found():
    """Test the generate_bias_dict function raises FileNotFoundError when bias.dat is not found."""

    # Create a temporary directory that does not contain the bias.dat file
    cache_dir = "/tmp/non_existent_cache_dir"

    with pytest.raises(FileNotFoundError) as exc_info:
        _ = generate_bias_dict(cache_dir=cache_dir)
        assert str(exc_info) == f"The bias.dat file was not found in the cache directory: {cache_dir}"


def test_debias():
    """Test the debias function to ensure it correctly debiases RA and Dec coordinates."""

    # Generate a bias dictionary for testing
    cache_dir = pooch.os_cache("layup")
    bias_dict = generate_bias_dict(cache_dir=cache_dir)

    # Test data
    ra = 10.0  # Right Ascension in degrees
    dec = 20.0  # Declination in degrees
    epoch = 2451545.0  # Julian date for J2000.0
    catalog = "PPM"  # Example catalog name

    # Call the debias function
    debiased_ra, debiased_dec = debias(ra, dec, epoch, catalog, bias_dict)

    # Check that the returned values are not None
    assert debiased_ra is not None, "Debiased RA should not be None."
    assert debiased_dec is not None, "Debiased Dec should not be None."

    # Check that the returned values are different from the input values (indicating debiasing occurred)
    assert debiased_ra != ra or debiased_dec != dec, "Debiased values should differ from input values."
