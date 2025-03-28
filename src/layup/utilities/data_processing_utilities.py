from concurrent.futures import ProcessPoolExecutor
import numpy as np

from layup.utilities.layup_configs import LayupConfigs
from sorcha.ephemeris.simulation_geometry import barycentricObservatoryRates
from sorcha.ephemeris.simulation_parsing import Observatory as SorchaObservatory
from sorcha.ephemeris.simulation_setup import furnish_spiceypy
import spiceypy as spice

""" A module for utilities useful for processing data in structured numpy arrays """


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
    block_size = max(1, int(len(data) / n_workers))
    # Create a list of tuples of the form (start, end) where start is the starting index of the block
    # and end is the last index of the block + 1.
    blocks = [(i, min(i + block_size, len(data))) for i in range(0, len(data), block_size)]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Create a future applying the function to each block of data
        futures = [executor.submit(func, data[start:end], **kwargs) for start, end in blocks]
        # Concatenate all processed blocks together as our final result
        return np.concatenate([future.result() for future in futures])


class LayupObservatory(SorchaObservatory):
    """
    A wrapper around Sorcha's Observatory class to provide additional functionality for Layup.
    """

    def __init__(self):
        # Get Layup configs
        config = LayupConfigs()

        # A simple class to mimic the arguments processed by Sorcha's observatory class
        class FakeSorchaArgs:
            def __init__(self):
                # Sorcha allows this argument to be None, so simply use that here
                self.ar_data_file_path = None

        # Furnish spiceypy kernels used for calculating barycentric positions
        furnish_spiceypy(FakeSorchaArgs(), config.auxiliary)
        super().__init__(FakeSorchaArgs(), config.auxiliary)

        # A cache of barycentric positions for observatories of the form {obscode: {et: (x, y, z)}}
        self.cached_obs = {}

    def obscodes_to_barycentric(self, data, fail_on_missing=False):
        """
        Takes a structured array of observations and returns the barycentric positions and velocites
        of the observatories.

        This assumes that data must have a column 'et' representing the ephemeris time of each
        observation in TDB.

        Parameters
        ----------
        data : numpy structured array
            The data to process.
        fail_on_missing : bool, optional
            If True, raise an error if we can't compute the barycentric position of an observatory.
            If False, return NaNs for the barycentric position of the observatory.

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
            coords = self.ObservatoryXYZ[obscode]
            if coords is None or None in coords or np.isnan(coords).any():
                # The observatory does not have a fixed position, so don't try to calculate barycentric coordinates
                # TODO most of the the time this is a moving observatory, and we should handle that case
                if fail_on_missing:
                    raise ValueError(f"Observatory {obscode} does not have a known fixed position.")
                bary_obs_pos, bary_obs_vel = [np.nan] * 3, [np.nan] * 3
            else:
                # Since the observatory has a known position, we can calculate the barycentric coordinates
                # at the observed epoch
                et = row["et"]
                if obscode not in self.cached_obs:
                    self.cached_obs[obscode] = {}
                try:
                    # Calculate the barycentric position and velocity of the observatory or fetch
                    # it from the cache if it has already been calculated
                    bary_obs_pos, bary_obs_vel = self.cached_obs[obscode].setdefault(
                        et, barycentricObservatoryRates(et, obscode, self)
                    )
                except Exception as e:
                    if fail_on_missing:
                        raise ValueError(
                            f"Error calculating barycentric coordinates for {obscode} at et: {et} from obstime: {row['obstime']} {e} "
                        )
                    bary_obs_pos, bary_obs_vel = [np.nan] * 3, [np.nan] * 3
            # Create a structured array for our barycentric coordinates with appropriate dtypes.
            x, y, z = bary_obs_pos[0], bary_obs_pos[1], bary_obs_pos[2]
            vx, vy, vz = bary_obs_vel[0], bary_obs_vel[1], bary_obs_vel[2]
            output_dtype = [
                ("x", "<f8"),
                ("y", "<f8"),
                ("z", "<f8"),
                ("vx", "<f8"),
                ("vy", "<f8"),
                ("vz", "<f8"),
            ]
            res.append(np.array((x, y, z, vx, vy, vz), dtype=output_dtype))

        # Combine all of our resutls into a single structured array
        return np.squeeze(np.array(res)) if len(res) > 1 else res[0]
