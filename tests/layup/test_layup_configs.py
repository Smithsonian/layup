import pytest
from layup.utilities.data_utilities_for_tests import get_config_setups_filepath
from layup.utilities.layup_configs import (
    LayupConfigs,
    AuxiliaryConfigs,
)


correct_auxciliary_URLs = {
    "de440s.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp",
    "earth_200101_990827_predict.bpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_200101_990827_predict.bpc",
    "earth_620120_240827.bpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_620120_240827.bpc",
    "earth_latest_high_prec.bpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc",
    "linux_p1550p2650.440": "https://ssd.jpl.nasa.gov/ftp/eph/planets/Linux/de440/linux_p1550p2650.440",
    "sb441-n16.bsp": "https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/asteroids_de441/sb441-n16.bsp",
    "naif0012.tls": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
    "ObsCodes.json.gz": "https://minorplanetcenter.net/Extended_Files/obscodes_extended.json.gz",
    "pck00010.pck": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc",
    "debias_hires2018.tgz": "ftp://ssd.jpl.nasa.gov/pub/ssd/debias/debias_hires2018.tgz",
}
correct_auxciliary_filenames = [
    "de440s.bsp",
    "earth_200101_990827_predict.bpc",
    "earth_620120_240827.bpc",
    "earth_latest_high_prec.bpc",
    "linux_p1550p2650.440",
    "sb441-n16.bsp",
    "naif0012.tls",
    "meta_kernel.txt",
    "ObsCodes.json.gz",
    "ObsCodes.json",
    "pck00010.pck",
    "debias_hires2018.tgz",
    "bias.dat",
]

##################################################################################################################################

# Layup Configs test


def test_layup_configs():
    """
    tests that LayupConfigs reads in config file correctly
    """
    # general test to make sure, overall, everything works. checks just one file: "default_config.ini"

    config_file_location = get_config_setups_filepath("default_config.ini")
    test_configs = LayupConfigs(config_file_location)
    # check each section to make sure you get what you expect
    assert correct_auxciliary_URLs == test_configs.auxiliary.__dict__["urls"]
    assert correct_auxciliary_filenames == test_configs.auxiliary.__dict__["data_file_list"]


##################################################################################################################################

# auxiliary config test


@pytest.mark.parametrize(
    "file",
    [
        "planet_ephemeris",
        "earth_predict",
        "earth_historical",
        "jpl_planets",
        "leap_seconds",
        "observatory_codes_compressed",
        "orientation_constants",
    ],
)
def test_auxiliary_config_url_given_filename_not(file):
    """Users can update the filenames in the config file as desired, but if they
    update the URL for a file, they must also update the filename for that file
    as well. This test checks that the user cannot update just the URL."""
    aux_configs = {file + "_url": "new_url"}
    with pytest.raises(SystemExit) as error_text:
        AuxiliaryConfigs(**aux_configs)
    assert error_text.value.code == f"ERROR: url for {file} given but filename for {file} not given"


@pytest.mark.parametrize(
    "file",
    [
        "planet_ephemeris",
        "earth_predict",
        "earth_historical",
        "jpl_planets",
        "leap_seconds",
        "observatory_codes_compressed",
        "orientation_constants",
    ],
)
def test_auxiliary_config_making_url_none(file):
    aux_configs = {file: "new_filename"}

    test_configs = AuxiliaryConfigs(**aux_configs)
    assert getattr(test_configs, file + "_url") == None
