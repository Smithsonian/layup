import gzip
import json
import logging
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from importlib.resources import files

import numpy as np
import requests
import spiceypy as spice
from sorcha.ephemeris.simulation_geometry import barycentricObservatoryRates
from sorcha.ephemeris.simulation_parsing import Observatory as SorchaObservatory
from sorcha.ephemeris.simulation_setup import furnish_spiceypy

from layup.constants import AU_KM
from layup.routines import FitResult
from layup.utilities.layup_configs import LayupConfigs

""" A module for utilities useful for processing data in structured numpy arrays """

# Start worker processes with "spawn" rather than the platform default.
#
# On Linux the default start method is "fork", which clones the whole parent
# process -- including any mutexes other threads hold at the instant of the
# fork, in their *locked* state.  layup imports JAX (via layup.convert ->
# orbit_conversion) at module load, and JAX/XLA runs background threads, so a
# forked worker can inherit a permanently-locked JAX mutex and deadlock the
# first time it touches it.  This is the cause of the ubuntu CI hangs in
# orbit-fit/convert workflows with num_workers > 1 (see issues #256 and #302).
#
# "spawn" launches a fresh interpreter per worker, inheriting none of the
# parent's locks or threads, so the deadlock cannot occur.  macOS already
# defaults to "spawn" (which is why the hang was Linux-only); pinning it here
# makes every platform behave the same.  ("forkserver" would also work and is
# cheaper, but "spawn" is the most portable and matches the macOS default.)
_MP_CONTEXT = multiprocessing.get_context("spawn")

logger = logging.getLogger(__name__)


def write_fallback_obscodes():
    """Decompress the observatory-codes file bundled with layup to a plain JSON
    file and return its path.

    The MPC observatory-codes file is normally downloaded from
    minorplanetcenter.net on first use.  When that download fails (e.g. the MPC
    server is unreachable, as has happened on CI runners), we fall back to the
    copy shipped in ``layup/data/ObsCodes.json.gz`` instead of failing the run.
    Observatory codes change rarely, so a slightly stale fallback is far better
    than a hard failure.  Returns a path suitable for ``Observatory``'s
    ``oc_file`` argument (which reads the decompressed JSON directly).
    """
    compressed = files("layup.data").joinpath("ObsCodes.json.gz").read_bytes()
    dest = os.path.join(tempfile.gettempdir(), "layup_obscodes_fallback.json")
    with open(dest, "wb") as f:
        f.write(gzip.decompress(compressed))
    return dest


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

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=_MP_CONTEXT) as executor:
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
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=_MP_CONTEXT) as executor:
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
    orbit_para=None,
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
    if orbit_para is None:
        orbit_para = ["x", "y", "z", "xdot", "ydot", "zdot"]

    if isinstance(fit_result_row, np.ndarray) and fit_result_row.shape == (1,):
        fit_result_row = fit_result_row[0]

    res = FitResult()

    if orbit_colm_flag:
        res.csq = fit_result_row["csq"]  # The chi-squared value of the fit
        res.ndof = fit_result_row["ndof"]  # The number of degrees of freedom
        # The number of iterations used during the fitting process.
        res.niter = fit_result_row["niter"]

    # The state vector of the fit result
    res.state = [fit_result_row[param] for param in orbit_para]
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

        try:
            super().__init__(FakeSorchaArgs(cache_dir), config.auxiliary)
        except (requests.exceptions.RequestException, json.JSONDecodeError) as exc:
            # Loading the MPC observatory-codes file failed. Either the download
            # itself errored (server unreachable, timeout, etc.), or it
            # "succeeded" but produced an empty/corrupt file that does not parse
            # as JSON -- both have been seen as transient CI failures. Either
            # way, fall back to the copy bundled with layup rather than failing
            # the run over a transient outage.
            logger.warning(
                "Could not load observatory codes from the MPC (%s); "
                "falling back to the copy bundled with layup.",
                exc,
            )
            super().__init__(
                FakeSorchaArgs(cache_dir),
                config.auxiliary,
                oc_file=write_fallback_obscodes(),
            )

        # A cache of barycentric positions for observatories of the form {obscode: {et: (x, y, z)}}
        self.cached_obs = {}

        # Optional user-supplied geocentric observer velocities (km/s, ICRF),
        # keyed by the same per-epoch cache key as ObservatoryXYZ (issue #147).
        # Empty unless ADES vel1/vel2/vel3 columns are provided for a moving
        # observer; _barycentric_moving_observatory falls back to Earth's
        # velocity for keys that are absent here.
        self.ObservatoryVel = {}

    def convert_to_geocentric(self, obs_location: dict) -> tuple:
        """Convert an observatory's parallax constants to geocentric coordinates.

        This overrides Sorcha's ``Observatory.convert_to_geocentric``, which gates
        on the truthiness of the parallax constants
        (``obs_location.get("sin", False)``). A constant that is legitimately
        ``0.0`` -- e.g. the geocenter (codes 500/244/248: Longitude=cos=sin=0),
        Greenwich (code 000: Longitude=0), or an equatorial station such as Quito
        (code 782: sin=0) -- is falsy, so the base class wrongly reports the
        observatory as having no fixed position. Layup then routes it to the
        moving-observatory path and raises "invalid coordinates" for plain MPC
        input that carries no per-observation position (issue #286).

        We instead test for the *presence* of the constants. Codes that have no
        position keys at all (roving observer 247, space telescopes WISE/TESS/HST,
        ...) still return ``(None, None, None)`` and are correctly routed to the
        ADES per-observation position path. The geocenter resolves to a (0, 0, 0)
        offset from Earth's center, which is exactly right.

        Parameters
        ----------
        obs_location : dict
            Dictionary with Longitude and the sin/cos of the observatory latitude.

        Returns
        -------
        tuple
            Geocentric position (x, y, z), or (None, None, None) when the
            observatory has no fixed position.
        """
        longitude = obs_location.get("Longitude")
        cos = obs_location.get("cos")
        sin = obs_location.get("sin")
        if longitude is not None and cos is not None and sin is not None:
            longitude_rad = longitude * np.pi / 180.0
            return (cos * np.cos(longitude_rad), cos * np.sin(longitude_rad), sin)
        return (None, None, None)

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
            # The observatory does not have a fixed position, so don't try to calculate barycentric coordinates.
            # A non-fixed observatory must carry its reference frame ('sys'), center ('ctr') and the three
            # position components ('pos1'/'pos2'/'pos3') in the data.
            for field in ("sys", "ctr", "pos1", "pos2", "pos3"):
                if field not in data.dtype.names:
                    raise ValueError(
                        f"The data must have a '{field}' field for non-fixed position observatory {obscode}."
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

            # Convert the coordinates to km if they are in AU
            if data["sys"] == "ICRF_AU":
                coords *= AU_KM

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

            # Optionally cache a user-supplied observer velocity. ADES permits the
            # velocity columns (vel1/vel2/vel3) only for space-based observers, so
            # we read them here, inside the moving-observer branch, reusing the
            # sys/ctr already validated for the position (issue #147).
            self._populate_observatory_velocity(obscode_cache_key, data)
        return obscode_cache_key

    def _populate_observatory_velocity(self, obscode_cache_key, data):
        """Cache an optional user-supplied geocentric observer velocity (km/s).

        ADES allows optional velocity columns (vel1/vel2/vel3) alongside the
        position of a space-based observer (issue #147). They share the
        position's reference frame (``sys``) and center (``ctr``), both already
        validated for the position in :meth:`populate_observatory`. We convert to
        km/s and store under the same per-epoch cache key as the position;
        :meth:`_barycentric_moving_observatory` then adds Earth's barycentric
        velocity to it. Velocity is optional: if the columns are absent or NaN we
        leave the cache untouched and fall back to Earth's velocity.

        Parameters
        ----------
        obscode_cache_key : str
            The per-epoch cache key the position was stored under.
        data : numpy structured array
            The observation row (already known to be a moving observer).
        """
        vel_fields = ("vel1", "vel2", "vel3")
        if not all(field in data.dtype.names for field in vel_fields):
            return
        vel = np.array([data["vel1"], data["vel2"], data["vel3"]], dtype=float)
        # Velocity is optional even when the columns exist (e.g. blank/NaN rows).
        if np.isnan(vel).any():
            return

        vel_km_s = self._obs_vel_to_km_s(vel, data["sys"])
        if obscode_cache_key in self.ObservatoryVel and not np.allclose(
            self.ObservatoryVel[obscode_cache_key], vel_km_s
        ):
            raise ValueError(
                f"Observatory velocity reported inconsistently at the same epoch for "
                f"cache key {obscode_cache_key}: previously "
                f"{self.ObservatoryVel[obscode_cache_key]}, now {vel_km_s} (km/s)."
            )
        self.ObservatoryVel[obscode_cache_key] = vel_km_s

    @staticmethod
    def _obs_vel_to_km_s(vel, sys):
        """Convert an ADES OBS_VEL observer velocity to km/s.

        Unit convention (matches the SPICE km/s state that
        :meth:`_barycentric_moving_observatory` uses for Earth):

        * ``ICRF_KM`` -> km/s (no conversion; SPICE-native)
        * ``ICRF_AU`` -> au/day -> km/s

        ``sys`` has already been validated to one of these two values in
        :meth:`populate_observatory`.
        """
        if sys == "ICRF_AU":
            return vel * AU_KM / (24 * 60 * 60)
        return vel

    def _barycentric_moving_observatory(self, et, obscode_cache_key):
        """Barycentric position and velocity (km, km/s) of a moving observatory.

        A fixed ground station's ObservatoryXYZ entry holds dimensionless
        parallax constants in the Earth-fixed frame, so
        barycentricObservatoryRates rotates it to J2000 and scales it by the
        Earth radius. A moving (e.g. space-based) observatory is different: its
        position is supplied per-observation as an explicit geocentric vector
        that populate_observatory has already stored in km and in the
        J2000/ICRF frame (from the sys/ctr/pos1..3 columns). It must therefore
        be added to Earth's barycentric position directly -- with no
        Earth-rotation and no Earth-radius scaling, both of which
        barycentricObservatoryRates would apply and which would misplace the
        observatory by a factor of the Earth radius (~6378x).

        The observatory's own geocentric velocity is used when the user supplies
        it via the ADES vel1/vel2/vel3 columns (cached in ObservatoryVel, km/s,
        issue #147): the barycentric velocity is then Earth's velocity plus the
        geocentric observer velocity, mirroring the position. When no velocity is
        supplied we fall back to Earth's barycentric velocity alone; the
        satellite's orbital velocity (~km/s) is then a small correction that only
        enters through aberration.
        """
        posvel, _ = spice.spkezr("EARTH", et, "J2000", "NONE", "SSB")
        earth_pos = np.array(posvel[0:3])  # km, J2000
        earth_vel = np.array(posvel[3:6])  # km/s, J2000
        obs_geocentric = np.array(self.ObservatoryXYZ[obscode_cache_key])  # km, J2000

        bary_pos = earth_pos + obs_geocentric
        obs_geocentric_vel = self.ObservatoryVel.get(obscode_cache_key)  # km/s or None
        if obs_geocentric_vel is not None:
            return bary_pos, earth_vel + obs_geocentric_vel
        return bary_pos, earth_vel

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
            # it from the cache if it has already been calculated. A per-epoch cache
            # key (obscode != key) marks a moving observatory whose position was
            # supplied in the data; it must not go through the fixed-station transform.
            if obscode_cache_key == obscode:
                bary = barycentricObservatoryRates(et, obscode_cache_key, self)
            else:
                bary = self._barycentric_moving_observatory(et, obscode_cache_key)
            bary_obs_pos, bary_obs_vel = self.cached_obs[obscode_cache_key].setdefault(et, bary)

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

    if "FORMAT" in data.dtype.names:
        # Find first valid format in the data
        for fmt in data["FORMAT"]:
            if fmt in ["BCART", "BCART_EQ", "BCOM", "BKEP", "CART", "COM", "KEP"]:
                return fmt
        logger.error("Data does not contain valid orbit format")
        raise ValueError("Data does not contain valid orbit format")
    else:
        logger.error("Data does not contain 'FORMAT' column")
        raise ValueError("Data does not contain 'FORMAT' column")


def skyplane_cov_to_radec_cov(cov_xx, cov_xy, cov_yy):
    """
    Convert a 2x2 sky-plane covariance into an on-sky error ellipse.

    The covariance is expressed in the local orthonormal tangent-plane basis
    used by predict, whose axes are the unit vectors in the directions of
    increasing RA (a great circle on the sky, i.e. already scaled by cos(dec))
    and increasing Dec. The error ellipse is therefore the eigen-decomposition
    of the 2x2 matrix; no cos(dec) scaling is applied (the input is already an
    on-sky covariance, not a covariance in RA/Dec coordinates).

    Parameters
    ----------
    cov_xx: numpy array
        (x, x) entry of the sky-plane covariance (radians^2); x is the
        great-circle RA direction.
    cov_xy: numpy array
        (x, y) entry of the sky-plane covariance (radians^2).
    cov_yy: numpy array
        (y, y) entry of the sky-plane covariance (radians^2); y is the Dec
        direction.

    Returns
    -----------
    numpy array
        semi-major axis of the error ellipse (arcsec)
    numpy array
        semi-minor axis of the error ellipse (arcsec)
    numpy array
        position angle of the major axis (degrees, North through East)
    """
    rad_to_arcsec = (180.0 / np.pi) * 3600.0

    # Eigenvalues of the symmetric 2x2 [[xx, xy], [xy, yy]].
    trace = cov_xx + cov_yy
    disc = np.sqrt(np.power(cov_xx - cov_yy, 2.0) + np.power(2.0 * cov_xy, 2.0))
    lambda_major = 0.5 * (trace + disc)
    lambda_minor = 0.5 * (trace - disc)

    # Guard against a tiny negative minor eigenvalue from floating-point error.
    a = np.sqrt(np.maximum(lambda_major, 0.0)) * rad_to_arcsec
    b = np.sqrt(np.maximum(lambda_minor, 0.0)) * rad_to_arcsec

    # Angle of the major axis from the RA (x) axis, reported North through East.
    PA = 90.0 - 0.5 * np.arctan2(2.0 * cov_xy, cov_xx - cov_yy) * 180.0 / np.pi

    return a, b, PA
