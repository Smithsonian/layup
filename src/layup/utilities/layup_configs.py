from dataclasses import dataclass, field
import configparser
import sys


@dataclass
class AuxiliaryConfigs:
    """Data class for holding auxiliary section configuration file keys and validating them."""

    naif_base_url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"
    ssd_base_url = "https://ssd.jpl.nasa.gov/ftp/eph"

    planet_ephemeris: str = "de440s.bsp"
    """filename of planet_ephemeris"""
    planet_ephemeris_url: str = f"{naif_base_url}/spk/planets/de440s.bsp"
    """url for planet_ephemeris"""

    earth_predict: str = "earth_200101_990827_predict.bpc"
    """filename of earth_predict"""
    earth_predict_url: str = f"{naif_base_url}/pck/earth_200101_990827_predict.bpc"
    """url for earth_predict"""

    earth_historical: str = "earth_620120_240827.bpc"
    """filename of earth_histoical"""
    earth_historical_url: str = f"{naif_base_url}/pck/earth_620120_240827.bpc"
    """url for earth_historical"""

    earth_high_precision: str = "earth_latest_high_prec.bpc"
    """filename of earth_high_precision"""
    earth_high_precision_url: str = f"{naif_base_url}/pck/earth_latest_high_prec.bpc"
    """url of earth_high_precision"""

    jpl_planets: str = "linux_p1550p2650.440"
    """filename of jpl_planets"""
    jpl_planets_url: str = f"{ssd_base_url}/planets/Linux/de440/linux_p1550p2650.440"
    """url of jpl_planets"""

    jpl_small_bodies: str = "sb441-n16.bsp"
    """filename of jpl_small_bodies"""
    jpl_small_bodies_url: str = f"{ssd_base_url}/small_bodies/asteroids_de441/sb441-n16.bsp"
    """url of jpl_small_bodies"""

    leap_seconds: str = "naif0012.tls"
    """filename of leap_seconds"""
    leap_seconds_url: str = f"{naif_base_url}/lsk/naif0012.tls"
    """url of leap_seconds"""

    meta_kernel: str = "meta_kernel.txt"
    """filename of meta_kernel"""

    observatory_codes: str = "ObsCodes.json"
    """filename of observatory_codes (uncompressed)"""

    observatory_codes_compressed: str = "ObsCodes.json.gz"
    """filename of observatory_codes as compressed file"""
    observatory_codes_compressed_url: str = (
        "https://minorplanetcenter.net/Extended_Files/obscodes_extended.json.gz"
    )
    """url of observatory_codes_compressed"""

    orientation_constants: str = "pck00010.pck"
    """filename of observatory_constants"""
    orientation_constants_url: str = f"{naif_base_url}/pck/pck00010.tpc"
    """url of observatory_constants"""

    data_file_list: list[str] = field(default_factory=list)
    """convenience list of all the file names"""

    urls: dict = field(default_factory=dict)
    """dictionary of {filename: url}"""

    data_files_to_download: list[str] = field(default_factory=list)
    """list of files that need to be downloaded"""

    ordered_kernel_files: list[str] = field(default_factory=list)
    """list of kernels ordered from least to most precise - used to assemble meta_kernel file"""

    registry: list[str] = field(default_factory=dict)
    """the Pooch registry to define which files will be tracked and retrievable"""

    @property
    def default_url(self):
        """returns a dictionary of the default urls used in this version of layup"""
        return {
            "planet_ephemeris": self.__class__.planet_ephemeris_url,
            "earth_predict": self.__class__.earth_predict_url,
            "earth_historical": self.__class__.earth_historical_url,
            "earth_high_precision": self.__class__.earth_high_precision_url,
            "jpl_planets": self.__class__.jpl_planets_url,
            "jpl_small_bodies": self.__class__.jpl_small_bodies_url,
            "leap_seconds": self.__class__.leap_seconds_url,
            "observatory_codes_compressed": self.__class__.observatory_codes_compressed_url,
            "orientation_constants": self.__class__.orientation_constants_url,
        }

    @property
    def default_filenames(self):
        """returns a dictionary of the default filenames used in this version of layup"""
        return {
            "planet_ephemeris": self.__class__.planet_ephemeris,
            "earth_predict": self.__class__.earth_predict,
            "earth_historical": self.__class__.earth_historical,
            "earth_high_precision": self.__class__.earth_high_precision,
            "jpl_planets": self.__class__.jpl_planets,
            "jpl_small_bodies": self.__class__.jpl_small_bodies,
            "leap_seconds": self.__class__.leap_seconds,
            "meta_kernel": self.__class__.meta_kernel,
            "observatory_codes_compressed": self.__class__.observatory_codes_compressed,
            "orientation_constants": self.__class__.orientation_constants,
        }

    def __post_init__(self):
        """create lists of files and validate the auxiliary configs after initialization."""
        self._create_lists_auxiliary_configs()
        self._validate_auxiliary_configs()

    def _validate_auxiliary_configs(self):
        """
        validates the auxiliary config attributes after initialization.
        """
        for file in self.default_filenames:
            if file != "meta_kernel":
                if (
                    self.default_filenames[file] == getattr(self, file)
                    and getattr(self, file + "_url") != self.default_url[file]
                ):
                    sys.exit(f"ERROR: url for {file} given but filename for {file} not given")

                elif (
                    self.default_filenames[file] != getattr(self, file)
                    and getattr(self, file + "_url") == self.default_url[file]
                ):
                    setattr(self, file + "_url", None)

    def _create_lists_auxiliary_configs(self):
        """
        creates lists of the auxiliary config attributes after initialization.

        Returns
        ----------
        None
        """

        self.urls = {
            self.planet_ephemeris: self.planet_ephemeris_url,
            self.earth_predict: self.earth_predict_url,
            self.earth_historical: self.earth_historical_url,
            self.earth_high_precision: self.earth_high_precision_url,
            self.jpl_planets: self.jpl_planets_url,
            self.jpl_small_bodies: self.jpl_small_bodies_url,
            self.leap_seconds: self.leap_seconds_url,
            self.observatory_codes_compressed: self.observatory_codes_compressed_url,
            self.orientation_constants: self.orientation_constants_url,
        }

        self.data_file_list = [
            self.planet_ephemeris,
            self.earth_predict,
            self.earth_historical,
            self.earth_high_precision,
            self.jpl_planets,
            self.jpl_small_bodies,
            self.leap_seconds,
            self.meta_kernel,
            self.observatory_codes_compressed,
            self.observatory_codes,
            self.orientation_constants,
        ]

        self.data_files_to_download = [
            self.planet_ephemeris,
            self.earth_predict,
            self.earth_historical,
            self.earth_high_precision,
            self.jpl_planets,
            self.jpl_small_bodies,
            self.leap_seconds,
            self.observatory_codes_compressed,
            self.orientation_constants,
        ]

        self.ordered_kernel_files = [
            self.leap_seconds,
            self.earth_historical,
            self.earth_predict,
            self.orientation_constants,
            self.planet_ephemeris,
            self.earth_high_precision,
        ]

        self.registry = {data_file: None for data_file in self.data_file_list}


@dataclass
class LayupConfigs:
    """Dataclass which stores configuration file keywords in dataclasses."""

    auxiliary: AuxiliaryConfigs = None
    """auxiliaryConfigs dataclass which stores the keywords from the AUXILIARY section of the config file."""

    # this __init__ overrides a dataclass's inbuilt __init__ because we want to populate this from a file, not explicitly ourselves
    def __init__(self, config_file_location=None):
        if config_file_location:  # if a location to a config file is supplied...
            config_object = configparser.ConfigParser()  # create a ConfigParser object
            config_object.read(config_file_location)  # and read the whole config file into it
            self._read_configs_from_object(
                config_object
            )  # now we call a function that populates the class attributes
        else:
            # if a file is not supplied the config class will be populated with the default values.
            self._populate_configs_class_with_default()

    def _read_configs_from_object(self, config_object):
        """
        function that populates the class attributes

        Parameters
        -----------
        config_object: ConfigParser object
            ConfigParser object that has the config file read into it

        Returns
        ----------
        None

        """

        # list of sections and corresponding config file
        section_list = {
            "AUXILIARY": AuxiliaryConfigs,
        }
        # when adding new sections in config file this general function needs the name of the section in uppercase
        # to be the same as the attributes defined above in lowercase e.g. section INPUT has attribute input
        # general function that reads in config file sections into there config dataclasses
        for section, config_section in section_list.items():
            if config_object.has_section(section):
                extra_args = {}
                # example for using extra args in a config section
                if section == "OUTPUT":
                    extra_args["example"] = "example"
                section_dict = dict(config_object[section])
                config_instance = config_section(**section_dict, **extra_args)

            else:
                config_instance = (
                    config_section()
                )  # if section not in config file take default values, This means if a blank config file is read in it will populate class with defaults.
            section_key = section.lower()
            setattr(self, section_key, config_instance)

    def _populate_configs_class_with_default(self):
        """
        function that populates the class attributes with their default values

        Parameters
        -----------
        None

        Returns
        ----------
        None
        """
        section_list = {
            "AUXILIARY": AuxiliaryConfigs,
        }
        for section, config_section in section_list.items():
            config_instance = config_section()
            section_key = section.lower()
            setattr(self, section_key, config_instance)
