import pytest
from layup.utilities.dataUtilitiesForTests import get_config_setups_filepath
from layup.utilities.layupConfigs import (
    layupConfigs,
    auxiliaryConfigs,
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
    "pck00010.pck",
]

##################################################################################################################################

# Layup Configs test


def test_layupConfigs():
    """
    tests that sorchaConfigs reads in config file correctly
    """
    # general test to make sure, overall, everything works. checks just one file: "Default_config_file.ini"

    config_file_location = get_config_setups_filepath("Default_config_file.ini")
    test_configs = layupConfigs(config_file_location)
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
        "observatory_codes",
        "orientation_constants",
    ],
)
def test_auxiliary_config_url_given_filename_not(file):

    aux_configs = {file + "_url": "new_url"}
    with pytest.raises(SystemExit) as error_text:
        test_configs = auxiliaryConfigs(**aux_configs)
    assert error_text.value.code == f"ERROR: url for {file} given but filename for {file} not given"


@pytest.mark.parametrize(
    "file",
    [
        "planet_ephemeris",
        "earth_predict",
        "earth_historical",
        "jpl_planets",
        "leap_seconds",
        "observatory_codes",
        "orientation_constants",
    ],
)
def test_auxiliary_config_making_url_none(file):
    aux_configs = {file: "new_filename"}

    test_configs = auxiliaryConfigs(**aux_configs)
    assert getattr(test_configs, file + "_url") == None
