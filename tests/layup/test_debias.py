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


@pytest.mark.parametrize("catalog", ["", None, "NOT_A_REAL_CATALOG"])
def test_debias_unknown_catalog_returns_unchanged(catalog):
    """A blank, None, or unrecognized star-catalog code has no bias model and must
    leave the astrometry unchanged instead of raising KeyError -- which would
    otherwise abort the whole object's fit on real/historical data (issue #401).
    These cases return before ``bias_dict`` is used, so an empty dict is fine.
    """
    ra, dec = 123.456, -7.89
    assert debias(ra, dec, 2451545.0, catalog, bias_dict={}) == (ra, dec)


def test_debias_known_catalog_absent_from_bias_dict_returns_unchanged():
    """A known catalog name that is not present in the supplied bias_dict also
    returns the astrometry unchanged (the pre-existing second guard)."""
    ra, dec = 50.0, 10.0
    assert debias(ra, dec, 2451545.0, "PPM", bias_dict={}) == (ra, dec)


def test_debias_accepts_catalog_code_like_a_name():
    """obs80 supplies the single-char catalog CODE (an MPC_CATALOGS value, e.g.
    ``"p"``), while ADES supplies the NAME (a key, e.g. ``"PPM"``). Both must apply
    the same bias correction -- otherwise debiasing silently no-ops on all obs80
    input (issue #409)."""
    cache_dir = pooch.os_cache("layup")
    bias_dict = generate_bias_dict(cache_dir=cache_dir)
    ra, dec, epoch = 10.0, 20.0, 2451545.0

    name, code = "PPM", MPC_CATALOGS["PPM"]  # "PPM" -> "p"
    ra_name, dec_name = debias(ra, dec, epoch, name, bias_dict)
    ra_code, dec_code = debias(ra, dec, epoch, code, bias_dict)

    # The code must resolve to the same correction as its name...
    assert (ra_code, dec_code) == (ra_name, dec_name)
    # ...and actually apply one (not silently return the input unchanged).
    assert (ra_code, dec_code) != (ra, dec)


def test_debias_epoch_and_nested_ordering():
    """Regression (no data files): the proper-motion term must use *years since
    J2000*, not the raw JD -- the raw JD extrapolates the proper motion over ~6700
    years and inflates the correction from sub-arcsec to ~10 arcsec, which turns
    every catalogued observation into a gross outlier. The bias table must also be
    read in NESTED healpix ordering. A synthetic one-pixel table pins both: 1"/yr
    of proper motion placed only at the NESTED pixel of (ra, dec), sampled 10 yr
    after J2000, must yield exactly 10" -- which requires both the J2000 epoch and
    the nested lookup.
    """
    import healpy as hp

    nside = 256
    ra, dec = 100.0, 20.0
    npix = hp.nside2npix(nside)
    nest_pix = hp.ang2pix(nside, ra, dec, nest=True, lonlat=True)
    zeros = np.zeros(npix)
    pm_dec = zeros.copy()
    pm_dec[nest_pix] = 1000.0  # 1000 mas/yr = 1 arcsec/yr, only at the nested pixel
    bias_dict = {"q": {"ra": zeros, "dec": zeros, "pm_ra": zeros, "pm_dec": pm_dec}}

    epoch = 2451545.0 + 10.0 * 365.25  # J2000 + 10 Julian years
    _, dec_deb = debias(ra, dec, epoch, "q", bias_dict)
    ddec_arcsec = (dec - dec_deb) * 3600.0
    assert abs(ddec_arcsec - 10.0) < 1e-6, f"expected 10 arcsec (10 yr x 1 arcsec/yr), got {ddec_arcsec}"


def test_debias_correction_is_arcsec_scale():
    """Real bias tables: the total catalog-bias correction at a modern epoch is a
    fraction of an arcsecond, not ~10 arcsec (the symptom of the J2000-epoch bug)."""
    cache_dir = pooch.os_cache("layup")
    bias_dict = generate_bias_dict(cache_dir=cache_dir)
    ra, dec = 100.0, 20.0
    epoch = 2459000.5  # ~2020
    for name in ("PPM", "UCAC4"):
        ra_deb, dec_deb = debias(ra, dec, epoch, name, bias_dict)
        shift = np.hypot((ra_deb - ra) * np.cos(np.deg2rad(dec)), dec_deb - dec) * 3600.0
        assert shift < 2.0, f"{name}: debias shift {shift:.2f} arcsec too large (epoch/ordering bug?)"
