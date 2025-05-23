import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sorcha.ephemeris.simulation_geometry import barycentricObservatoryRates
from sorcha.ephemeris.simulation_parsing import Observatory as SorchaObservatory
from sorcha.ephemeris.simulation_setup import furnish_spiceypy

from layup.routines import FitResult
from layup.utilities.layup_configs import LayupConfigs

""" A module for utilities useful for processing data in structured numpy arrays """

AU_M = 149597870700
AU_KM = AU_M / 1000.0

logger = logging.getLogger(__name__)


def process_data(data, n_workers, func, **kwargs):
    """
    Process a structured numpy array in parallel for a given function and keyword arguments

    Parameters
    ----------
    data : numpy structured array
        The data to process.
    n_workers : int
        The number of workers to use for parallel processing.
    func : function
        The function to apply to each block of data within parallel.
    **kwargs : dictionary
        Extra arguments to pass to the function.

    Returns
    -------
    res : numpy structured array
        The processed data concatenated from each function result
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be greater than 0, {n_workers} was provided.")

    if len(data) == 0:
        return data

    # Divide our data into blocks to be processed by each worker
    block_size = max(1, int(np.ceil(len(data) / n_workers)))
    # Create a list of tuples of the form (start, end) where start is the starting index of the block
    # and end is the last index of the block + 1.
    blocks = [(i, min(i + block_size, len(data))) for i in range(0, len(data), block_size)]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Create a future applying the function to each block of data
        futures = [executor.submit(func, data[start:end], **kwargs) for start, end in blocks]
        # Concatenate all processed blocks together as our final result
        return np.concatenate([future.result() for future in futures])


def process_data_by_id(data, n_workers, func, primary_id_column_name, **kwargs):
    """
    Process a structured numpy array in parallel for a given function and
    keyword arguments. Instead of distributing the data across all available workers
    it is expected that the data will contain a primary id column. The data will be
    split by the unique values in the primary id column and each block of data will
    be processed in parallel.

    Parameters
    ----------
    data : numpy structured array
        The data to process. Expected to contain a primary id column.
    n_workers : int
        The number of workers to use for parallel processing.
    func : function
        The function to apply to each block of data within parallel.
    **kwargs : dictionary
        Extra arguments to pass to the function.

    Returns
    -------
    res : numpy structured array
        The processed data concatenated from each function result
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be greater than 0, {n_workers} was provided.")

    #! Perhaps this should be None, or raise and exception that is caught by the
    #! caller. If we return `data`, the columns won't match the columns of the
    #! processed data.
    if len(data) == 0:
        return data

    kwargs["primary_id_column_name"] = primary_id_column_name
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Create a future applying the function to each block of data for a given object id
        futures = [
            executor.submit(func, data[data[primary_id_column_name] == id], **kwargs)
            for id in np.unique(data[primary_id_column_name])
        ]
        # Concatenate all processed blocks together as our final result
        return np.concatenate([future.result() for future in futures])


def get_cov_columns():
    """
    Get the covariance columns that are expected in the structured numpy array
    representing our orbit fit output result.

    Columns are a flattened version of the covariance matrix, which is a 6x6 matrix
    where the first first row and first 6 items are:

    [cov_00, cov_01, cov_02, cov_03, cov_04, cov_05]

    and the last row and last 6 items of the flattened matrix are:
    [cov_50, cov_51, cov_52, cov_53, cov_54, cov_55]

    Returns
    -------
    cov_columns : list[str]
        The covariance columns in the data.
    """
    # Get the covariance columns from the data
    return [f"cov_{i}_{j}" for i in range(6) for j in range(6)]


def has_cov_columns(data):
    """
    Check if the data has the expected covariance columns.

    Parameters
    ----------
    data : numpy structured array
        The data to check.

    Returns
    -------
    bool
        True if the data has covariance columns, False otherwise.
    """
    # Check if the data has the expected covariance columns
    return all(col in data.dtype.names for col in get_cov_columns())


def parse_cov(orbit_row, flatten=False):
    """
    Parse the covariance matrix from a structured numpy array representing our
    orbit fit output result.

    Parameters
    ----------
    orbit_row : numpy structured array
        The row of the structured array representing an orbit.
    flatten: bool, optional
        If True, return a flattened covariance matrix. If False, return a 6x6 covariance matrix.
        Default is False.
    Returns
    -------
    cov : numpy array
        The parsed covariance matrix.
    """
    if not has_cov_columns(orbit_row):
        raise ValueError("The row does not have the expected covariance columns.")
    # Construct the flattened covariance matrix from the columns of the fit result
    res = np.array([orbit_row[col] for col in get_cov_columns()])
    return res if flatten else res.reshape((6, 6))


def parse_fit_result(
    fit_result_row,
    orbit_colm_flag=True,
    orbit_para=[
        "x",
        "y",
        "z",
        "xdot",
        "ydot",
        "zdot",
    ],
):
    """
    Parse the initial guess data from a structured numpy array representing our
    orbit fit output result.

    Parameters
    ----------
    fit_result_row : numpy structured array
        The row of the structured array representing the orbit fit result.
    Returns
    -------
    res : FitResult
        The parsed fit result.
    """
    res = FitResult()

    if orbit_colm_flag == True:
        res.csq = fit_result_row["csq"]  # The chi-squared value of the fit
        res.ndof = fit_result_row["ndof"]  # The number of degrees of freedom
        # The number of iterations used during the fitting process.
        res.niter = fit_result_row["niter"]

    # The state vector of the fit result
    res.state = res.state = [fit_result_row[param] for param in orbit_para]
    # While orbitfit saves the epoch in MJD_TDB, internal calculations use JD_TDB
    res.epoch = fit_result_row["epochMJD_TDB"] + 2400000.5
    # Construct the flattened covariance matrix from the columns of the fit result
    cov = np.zeros(36)
    for i, col in enumerate(get_cov_columns()):
        try:
            # If there is no value in the input file that this row came from,
            # the fit_result_row[col] will be np.nan.
            if not np.isnan(fit_result_row[col]):
                cov[i] = fit_result_row[col]
            else:
                cov[i] = 0.0
        except ValueError:
            # If the `col` column isn't present in fit_result_row, we assign 0.0
            cov[i] = 0.0
    res.cov = cov

    return res


def create_chunks(reader, chunk_size):
    """For a given reader create a list of lists of object ids such that the total
    number of entries in the file for all object ids in a given list, will be
    less than the chunk size.

    Parameters
    ----------
    reader : ObjectDataReader
        The file reader object for the input file
    chunk_size : int
        The maximum number of rows to be included in a single list of ids

    Returns
    -------
    chunks : list[list[ObjIds]]
        A list of lists of object ids that can be passed to the reader's read_objects
        method.
    """
    # Force the reader to build the id table and id count dictionary
    reader._build_id_map()

    # Find all object ids with more rows than the max allowed number of rows.
    exceeds_id_list = []
    for k, v in reader.obj_id_counts.items():
        if v > chunk_size:
            exceeds_id_list.append(k)

    # Log an error if the any of the objects have more rows than the chunk size
    if exceeds_id_list:
        logger.error("The following objects have more rows than the max allowed number of rows.")
        for k in exceeds_id_list:
            logger.error(f"Object id {k} has {reader.obj_id_counts[k]} rows")
        raise ValueError("At least one object has more rows than the max allowed number of rows.")

    chunks = []
    obj_ids_in_chunk = []
    accumulator = 0

    # Loop over the object id counts dictionary
    for k, v in reader.obj_id_counts.items():
        # Check if the chunk size is exceeded, if so, save the current chunk and start a new chunk
        if accumulator + v > chunk_size:
            chunks.append(obj_ids_in_chunk)
            obj_ids_in_chunk = []
            accumulator = 0

        # Increase the accumulator and add the object id to the current chunk
        accumulator += v
        obj_ids_in_chunk.append(k)

    # Add the last chunk if it is not empty
    if obj_ids_in_chunk:
        chunks.append(obj_ids_in_chunk)

    return chunks


# Simple wrapper class to mimic the arguments expected by Sorcha methods.
class FakeSorchaArgs:
    def __init__(self, cache_dir=None):
        # Sorcha allows this argument to be None, so simply use that here
        self.ar_data_file_path = cache_dir


def layup_furnish_spiceypy(cache_dir):
    """A simple wrapper to furnish spiceypy kernels."""
    # A simple class to mimic the arguments processed by Sorcha's observatory class
    config = LayupConfigs()
    furnish_spiceypy(FakeSorchaArgs(cache_dir), config.auxiliary)


class LayupObservatory(SorchaObservatory):
    """
    A wrapper around Sorcha's Observatory class to provide additional functionality for Layup.
    """

    def __init__(self, cache_dir=None):
        """Create an instance of the LayupObservatory class.

        Parameters
        ----------
        cache_dir : str, optional
            The location of the cache directory containing the bootstrapped files.
            If the files or cache is not present, the files will be downloaded, by default None
        """

        # Get Layup configs
        config = LayupConfigs()

        # Furnish the spiceypy kernels
        layup_furnish_spiceypy(cache_dir)

        super().__init__(FakeSorchaArgs(cache_dir), config.auxiliary)

        # A cache of barycentric positions for observatories of the form {obscode: {et: (x, y, z)}}
        self.cached_obs = {}

    def create_obscode_cache_key(self, obscode, et):
        """
        Create a cache key for the observatory coordinates.

        Parameters
        ----------
        obscode : str
            The observatory code.
        et : float
            The ephemeris time.

        Returns
        -------
        str
            The cache key for the observatory coordinates.
        """
        return f"{obscode}_{et}"

    def populate_observatory(self, obscode, et, data):
        """
        Populate the observatory coordinates for a given observatory code and ephemeris time and
        provide the key that can be used to access the coordinates for the observatory in the cache
        at the given epoch. This is used to generalize the case where the observatory
        does not have a fixed position and the coordinates are provided in the data.

        Parameters
        ----------
        obscode : str
            The observatory code.
        et : float
            The ephemeris time.
        data : numpy structured array
            A row of the structured array of the orbit data to process.

        Returns
        -------
        obscode_cache_key : str
            The cache key for the observatory coordinates at the given epoch.
        """
        obscode_cache_key = obscode
        coords = self.ObservatoryXYZ.get(obscode, None)
        # Update the cached coordinates and obscode_cache_key for the case of a moving observatory
        if coords is None or None in coords or np.isnan(coords).any():
            obscode_cache_key = self.create_obscode_cache_key(obscode, et)
            # The observatory does not have a fixed position, so don't try to calculate barycentric coordinates
            if "sys" not in data.dtype.names:
                raise ValueError(
                    f"The data must have a 'sys' field for the reference frame of non-fixed position observatory {obscode}."
                )
            if "ctr" not in data.dtype.names:
                raise ValueError(
                    f"The data must have a 'ctr' field for non-fixed position observatory {obscode}."
                )
            if "pos1" not in data.dtype.names:
                raise ValueError(
                    f"The data must have a 'pos1' field for non-fixed position observatory {obscode}."
                )
            if "pos2" not in data.dtype.names:
                raise ValueError(
                    f"The data must have a 'pos2' field for non-fixed position observatory {obscode}."
                )
            if "pos3" not in data.dtype.names:
                raise ValueError(
                    f"The data must have a 'pos3' field for non-fixed postion observatory {obscode}."
                )
            coords = np.array([data["pos1"], data["pos2"], data["pos3"]])
            # If any of the coordinates are None or NaN, raise an error
            if coords is None or np.isnan(coords).any():
                raise ValueError(f"Observatory {obscode} has invalid coordinates at epoch {et}: {coords}")

            # Check if the coordinates are in a reference frame that we support.
            if data["sys"] not in ["ICRF_KM", "ICRF_AU"]:
                raise ValueError(
                    f"Observatory {obscode} has an unsupported reference frame {data['sys']} at epoch {et}. Please use ICRF_KM or ICRF_AU."
                )
            if data["ctr"] != 399:
                raise ValueError(
                    f"Observatory {obscode} has an unsupported center {data['ctr']}. Please use the 399 (Earth)."
                )

            if obscode_cache_key not in self.ObservatoryXYZ:
                # Store the coordinates in the ObservatoryXYZ dictionary to be read by barycentricObservatoryRates
                self.ObservatoryXYZ[obscode_cache_key] = coords
            else:
                # If the coordinates are not the same, raise an error
                if not np.allclose(self.ObservatoryXYZ[obscode_cache_key], coords):
                    raise ValueError(
                        f"Observatory {obscode} has different coordinates reported at the same epoch."
                        f"Coordinates at epoch {et} previously were {self.ObservatoryXYZ[obscode_cache_key]}, but are now {coords}."
                    )
            # Save the coordinates in the cache for the given obscode and epoch
            self.ObservatoryXYZ[obscode_cache_key] = coords
        return obscode_cache_key

    def obscodes_to_barycentric(self, data):
        """
        Takes a structured array of observations and returns the barycentric positions and velocites
        of the observatories.

        This assumes that data must have a column 'et' representing the ephemeris time of each
        observation in TDB.

        Parameters
        ----------
        data : numpy structured array
            The data to process.

        Returns
        -------
        res : numpy structured array
            Representing the barycentric positions and velocities of the observatories in the data (x,y,z,vx,vy,vz).
        """
        if "stn" not in data.dtype.names:
            raise ValueError("The data must have a 'stn' field.")

        res = []
        for row in data:
            obscode = row["stn"]
            if not isinstance(obscode, str):
                raise ValueError(
                    f"observatory code {obscode} is not a string and instead has type {type(obscode)}"
                )
            et = row["et"]

            # Check if the observatory code is valid and populate the observatory coordinates
            # in the cache if it is not already present (such as for a moving observatory)
            obscode_cache_key = self.populate_observatory(obscode, et, row)

            # Use the observatory position to calculate the barycentric coordinates at the observed epoch
            if obscode_cache_key not in self.cached_obs:
                self.cached_obs[obscode_cache_key] = {}

            # Calculate the barycentric position and velocity of the observatory or fetch
            # it from the cache if it has already been calculated
            bary_obs_pos, bary_obs_vel = self.cached_obs[obscode_cache_key].setdefault(
                et, barycentricObservatoryRates(et, obscode_cache_key, self)
            )

            # Create a structured array for our barycentric coordinates with appropriate dtypes.
            # Needed to adjust the units here.
            x, y, z = bary_obs_pos / AU_KM
            vx, vy, vz = bary_obs_vel * (24 * 60 * 60) / AU_KM
            output_dtype = [
                ("x", "<f8"),
                ("y", "<f8"),
                ("z", "<f8"),
                ("vx", "<f8"),
                ("vy", "<f8"),
                ("vz", "<f8"),
            ]
            res.append(np.array((x, y, z, vx, vy, vz), dtype=output_dtype))

        # Combine all of our results into a single structured array
        return np.squeeze(np.array(res)) if len(res) > 1 else res[0]


def get_format(data):
    """
    Get the orbit parameter format for this data.

    Parameters
    ----------
    data : numpy structured array
        The data to check.

    Returns
    -------
    str
        The format of the data.
    """

    if len(data) == 0:
        logger.error("Data is empty")
        raise ValueError("Data is empty")

    format = None

    if "FORMAT" in data.dtype.names:
        # Find first valid format in the data
        for format in data["FORMAT"]:
            if format in ["BCART", "BCART_EQ", "BCOM", "BKEP", "CART", "COM", "KEP"]:
                return format
        else:
            logger.error("Data does not contain valid orbit format")
            raise ValueError("Data does not contain valid orbit format")
    else:
        logger.error("Data does not contain 'FORMAT' column")
        raise ValueError("Data does not contain 'FORMAT' column")
