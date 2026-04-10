import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from layup.utilities.file_io.CSVReader import CSVDataReader
from layup.utilities.file_io.HDF5Reader import HDF5DataReader
from layup.utilities.data_processing_utilities import get_format

from layup.orbit_maths import (
    REQUIRED_COLUMN_NAMES,
    build_ephem_and_mus,
    build_planet_lines_cache,
    conic_lines_from_classical_conic,
    prepopulate_orbit_variants,
)
from layup.dash_ui import plotly_2D, plotly_3D, run_dash_app

logger = logging.getLogger(__name__)

DASH_THREAD = None


def build_fig_caches(
    rows: np.ndarray,
    orbit_format: str,
    input_plane: Literal["equatorial", "ecliptic"],
    input_origin: Literal["heliocentric", "barycentric"],
    special_rows: Optional[np.ndarray] = None,
    n_points: int = 500,
    r_max: float = 50.0,
    cache_dir: Optional[str] = None,
):
    """ """
    # get the assist ephem object and build epoch array
    logger.info(f"Building Assist ephemeris for planets")
    ephem, _, _ = build_ephem_and_mus(cache_dir)
    epochJD_center = float(rows["epochMJD_TDB"].astype(float)[0] + 2400000.5)

    # construct planet lines
    logger.info(f"Constructing planet orbit lines")
    planet_names = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    planet_lines_cache, planet_id = build_planet_lines_cache(
        ephem,
        epochJD_center,
        planet_names=planet_names,
        n_points=900,
    )

    # if a special file was provided, drop any rows from the main set whose ObjID
    # appears in the special set to avoid double-plotting the same orbit
    if special_rows is not None and special_rows.size > 0 and "ObjID" in rows.dtype.names:
        special_obj_ids = set(str(oid) for oid in special_rows["ObjID"])
        mask = np.array([str(oid) not in special_obj_ids for oid in rows["ObjID"]])
        n_dropped = int((~mask).sum())
        if n_dropped > 0:
            logger.info(
                f"Dropping {n_dropped} orbit(s) from main set whose ObjID also appears in the special file"
            )
            rows = rows[mask]

    # construct input orbits lines, scatters, and sun positions
    logger.info(f"Constructing input object orbit lines")
    conic_cache, lines_cache, sunpos_cache, pos_cache = prepopulate_orbit_variants(
        rows,
        orbit_format,
        input_plane=input_plane,
        input_origin=input_origin,
    )

    for key, conic in conic_cache.items():
        lines_cache[key] = conic_lines_from_classical_conic(
            conic,
            n_points=n_points,
            r_max=r_max,
        )

    # construct special orbit lines if a special file was provided
    special_conic_cache = None
    special_lines_cache = None
    special_ids = []
    if special_rows is not None and special_rows.size > 0:
        logger.info(f"Constructing special object orbit lines")
        special_conic_cache, special_lines_tmp, _, _ = prepopulate_orbit_variants(
            special_rows,
            orbit_format,
            input_plane=input_plane,
            input_origin=input_origin,
        )
        special_lines_cache = {}
        for key, conic in special_conic_cache.items():
            special_lines_cache[key] = conic_lines_from_classical_conic(
                conic,
                n_points=n_points,
                r_max=r_max,
            )
        # collect IDs from first available key
        first_key = next(iter(special_conic_cache))
        special_ids = [str(oid) for oid in special_conic_cache[first_key].obj_id]

    logger.info(
        f"Creating cache of all input object and/or planet lines in heliocentric+barycentric && equatorial+ecliptic frames"
    )
    # create caches of all figure combinations
    PANELS = ("XY", "XZ", "YZ")
    fig2d_cache = {}
    fig3d_cache = {}
    for key in conic_cache.keys():
        s_lines = special_lines_cache[key] if special_lines_cache else None
        s_canon = special_conic_cache[key] if special_conic_cache else None

        # 3D has few variants
        fig3d_cache[key] = plotly_3D(
            lines_cache[key],
            conic_cache[key],
            sun_xyz=sunpos_cache[key],
            planet_lines=planet_lines_cache[key],
            planet_id=planet_id,
            plot_sun=True,
            special_lines=s_lines,
            special_canon=s_canon,
            return_fig=True,
        )

        # 2D however has ref plane + origin + XY/XZ/YZ panel combinations + single or double panels
        # first do single panel variants
        for p in PANELS:
            fig2d_cache[(key[0], key[1], "single", p, None)] = plotly_2D(
                lines_cache[key],
                conic_cache[key],
                sun_xyz=sunpos_cache[key],
                planet_lines=planet_lines_cache[key],
                planet_id=planet_id,
                plot_sun=True,
                panel=p,
                special_lines=s_lines,
                special_canon=s_canon,
                return_fig=True,
            )

        # now do double panel variants:
        for pL in PANELS:
            for pR in PANELS:
                fig2d_cache[(key[0], key[1], "double", pL, pR)] = plotly_2D(
                    lines_cache[key],
                    conic_cache[key],
                    sun_xyz=sunpos_cache[key],
                    planet_lines=planet_lines_cache[key],
                    planet_id=planet_id,
                    plot_sun=True,
                    panels=(pL, pR),
                    special_lines=s_lines,
                    special_canon=s_canon,
                    return_fig=True,
                )

    return fig2d_cache, fig3d_cache, special_ids


def visualize_cli(
    input: str,
    num_orbs: int = 100,
    block_size: int = 10000,
    n_points: int = 500,
    r_max: float = 50.0,
    random: bool = False,
    cache_dir: Optional[str] = None,
    special: Optional[str] = None,
):
    """
    Create visualisation plots of a given set of input orbits from the command line

    Parameters
    -----------
    input : str
        Input file path

    num_orbs : int, optional (default=100)
        Number of orbits to plot at once

    block_size : int, optional (default=10000)
        Number of rows to read per time in the input file reader

    n_points : int, optional (default=500)
        Number of points sampled when constructing the line

    r_max : float, optional (default=50 au)
        Maximum distance to render hyperbolic orbits out to

    random : bool, optional (default=False)
        Flag to turn on/off random orbit plotting

    cache_dir : str, optional (default=None)
        Path to directory of cached auxiliary data

    special : str, optional (default=None)
        Path to a second orbit file whose orbits are highlighted in a distinct accent colour;
        regular orbits are greyed out when this is supplied
    """

    logger.info(f"Reading input file: {input}")
    # read in input
    input_file = Path(input)
    if not input_file.exists():
        logger.error(f"File not found: {input}")
        raise FileNotFoundError(input_file)

    # probe reader to get format
    logger.info(f"Probing input to infer orbit origin, reference plane, and format...")
    suffix = input_file.suffix.lower()
    if suffix == ".csv":
        probe_reader = CSVDataReader(
            input_file, format_column_name="FORMAT", required_column_names=["FORMAT"]
        )
    else:
        probe_reader = HDF5DataReader(
            input_file, format_column_name="FORMAT", required_column_names=["FORMAT"]
        )
    probe_rows = probe_reader.read_rows(block_start=0, block_size=100)
    if "FORMAT" in probe_rows.dtype.names:
        n = np.unique(probe_rows["FORMAT"])
        if n.size != 1:
            logger.error(f"Expected a single FORMAT in file, found {n}")
            raise ValueError(f"Expected a single FORMAT in file, found {n}")
    orbit_format = get_format(probe_rows)

    # determine input frame — default is heliocentric ecliptic (as per MPC/JPL convention)
    # BCART_EQ is the one exception: that format name explicitly encodes barycentric+equatorial
    input_format_infer = orbit_format
    if input_format_infer == "BCART_EQ":
        input_format_infer = "BCART"
        input_plane_infer = "equatorial"
        input_origin_infer = "barycentric"
        logger.info("FORMAT=BCART_EQ detected: using barycentric equatorial reference frame")
    else:
        input_plane_infer = "ecliptic"
        input_origin_infer = "heliocentric"
        logger.warning(
            f"Assuming heliocentric ecliptic input reference frame (as per MPC/JPL convention) for FORMAT={orbit_format}"
        )

    logger.info(f"Input orbit origin: {input_origin_infer}")
    logger.info(f"Input orbit reference plane: {input_plane_infer}")
    logger.info(f"Input orbit format: {input_format_infer}")

    # full reader with required columns
    logger.info(f"Reading full input file: {input}")
    required_cols = REQUIRED_COLUMN_NAMES[orbit_format]
    if suffix == ".csv":
        reader = CSVDataReader(input_file, format_column_name="FORMAT", required_column_names=required_cols)
    else:
        reader = HDF5DataReader(input_file, format_column_name="FORMAT", required_column_names=required_cols)
    rows = reader.read_rows(block_start=0, block_size=block_size)

    # make sure we actually have the requested number of orbits, else return all,
    # then either randomly sample that many or sample the first that many
    if num_orbs > rows.size:
        num_orbs = rows.size
        logger.warning(
            f"Requested {num_orbs} orbits, but only {rows.size} orbits in input. Capping to {rows.size}"
        )
        # raise Warning("--input-plane is required for CART/BCART formats unless FORMAT column encodes it (e.g. BCART_EQ)")
    if random:
        logger.info(f"Sampling {num_orbs} random orbits")
        rows = rows[np.random.choice(rows.size, size=num_orbs, replace=False)]
    else:
        logger.info(f"Sampling first {num_orbs} orbits")
        rows = rows[:num_orbs]

    # quick check to make sure orbit format didn't change somehow between probe and full read
    orbit_format_check = get_format(rows)
    if orbit_format_check != orbit_format:
        logger.error(f"FORMAT changed between probe and full read: {orbit_format} -> {orbit_format_check}")
        raise ValueError(
            f"FORMAT changed between probe and full read: {orbit_format} -> {orbit_format_check}"
        )

    # read optional special file
    special_rows = None
    if special is not None:
        logger.info(f"Reading special input file: {special}")
        special_file = Path(special)
        if not special_file.exists():
            logger.error(f"Special file not found: {special}")
            raise FileNotFoundError(special_file)
        special_suffix = special_file.suffix.lower()
        if special_suffix == ".csv":
            special_reader = CSVDataReader(
                special_file, format_column_name="FORMAT", required_column_names=required_cols
            )
        else:
            special_reader = HDF5DataReader(
                special_file, format_column_name="FORMAT", required_column_names=required_cols
            )
        special_rows = special_reader.read_rows(block_start=0, block_size=block_size)
        special_format = get_format(special_rows)
        if special_format != orbit_format:
            logger.error(
                f"Special file FORMAT ({special_format}) does not match main file FORMAT ({orbit_format})"
            )
            raise ValueError(
                f"Special file FORMAT ({special_format}) does not match main file FORMAT ({orbit_format})"
            )
        logger.info(f"Loaded {special_rows.size} special orbits from {special}")

    fig2d_cache, fig3d_cache, special_ids = build_fig_caches(
        rows=rows,
        orbit_format=orbit_format,
        input_plane=input_plane_infer,
        input_origin=input_origin_infer,
        special_rows=special_rows,
        n_points=n_points,
        r_max=r_max,
        cache_dir=cache_dir,
    )

    logger.info(f"Running Dash web app")
    run_dash_app(fig2d_cache, fig3d_cache, special_ids=special_ids)


def visualize_notebook(
    data: str | Path | np.ndarray,
    num_orbs: int = 100,
    block_size: int = 10000,
    n_points: int = 500,
    r_max: float = 50.0,
    random: bool = False,
    cache_dir: Optional[str] = None,
    special: Optional[str | Path | np.ndarray] = None,
):
    """
    Create visualisation plots of a given set of input orbits in a Jupyter notebook

    Parameters
    -----------
    data : str, Path, or numpy structured array
        Input file path or structured array with a FORMAT column

    num_orbs : int, optional (default=100)
        Number of orbits to plot at once

    block_size : int, optional (default=10000)
        Number of rows to read per time in the input file reader

    n_points : int, optional (default=500)
        Number of points sampled when constructing the line

    r_max : float, optional (default=50 au)
        Maximum distance to render hyperbolic orbits out to

    random : bool, optional (default=False)
        Flag to turn on/off random orbit plotting

    cache_dir : str, optional (default=None)
        Path to directory of cached auxiliary data
    """
    if isinstance(data, (str, Path)):
        special_path = str(special) if isinstance(special, (str, Path)) else None
        visualize_cli(
            str(data),
            num_orbs=num_orbs,
            block_size=block_size,
            n_points=n_points,
            r_max=r_max,
            random=random,
            cache_dir=cache_dir,
            special=special_path,
        )
    elif isinstance(data, np.ndarray):
        if data.dtype.names is None or "FORMAT" not in data.dtype.names:
            logger.error(
                "Structured array input must contain a FORMAT column, which must be one of: ['CART'. 'BCART', 'BCART_EQ', 'KEP', 'BKEP', 'COM', 'BCOM'])"
            )
            raise ValueError(
                "Structured array input must contain a FORMAT column, which must be one of: ['CART'. 'BCART', 'BCART_EQ', 'KEP', 'BKEP', 'COM', 'BCOM'])"
            )

        rows = data
        orbit_format = get_format(rows)

        # determine input frame — default is heliocentric ecliptic (as per MPC/JPL convention)
        # BCART_EQ is the one exception: that format name explicitly encodes barycentric+equatorial
        input_format_infer = orbit_format
        if input_format_infer == "BCART_EQ":
            input_format_infer = "BCART"
            input_plane_infer = "equatorial"
            input_origin_infer = "barycentric"
            logger.info("FORMAT=BCART_EQ detected: using barycentric equatorial reference frame")
        else:
            input_plane_infer = "ecliptic"
            input_origin_infer = "heliocentric"
            logger.warning(
                f"Assuming heliocentric ecliptic input reference frame (as per MPC/JPL convention) for FORMAT={orbit_format}"
            )

        logger.info(f"Input orbit origin: {input_origin_infer}")
        logger.info(f"Input orbit reference plane: {input_plane_infer}")
        logger.info(f"Input orbit format: {input_format_infer}")

        # make sure we actually have the requested number of orbits, else return all,
        # then either randomly sample that many or sample the first that many
        if num_orbs > rows.size:
            logger.warning(
                f"Requested {num_orbs} orbits, but only {rows.size} orbits in input. Capping to {rows.size}"
            )
            num_orbs = rows.size
            # raise Warning("--input-plane is required for CART/BCART formats unless FORMAT column encodes it (e.g. BCART_EQ)")
        if random:
            logger.info(f"Sampling {num_orbs} random orbits")
            rows = rows[np.random.choice(rows.size, size=num_orbs, replace=False)]
        else:
            logger.info(f"Sampling first {num_orbs} orbits")
            rows = rows[:num_orbs]

        # quick check to make sure orbit format didn't change somehow between probe and full read
        orbit_format_check = get_format(rows)
        if orbit_format_check != orbit_format:
            logger.error(
                f"FORMAT changed between probe and full read: {orbit_format} -> {orbit_format_check}"
            )
            raise ValueError(
                f"FORMAT changed between probe and full read: {orbit_format} -> {orbit_format_check}"
            )

        # handle special rows if provided as a numpy array
        special_rows = None
        if isinstance(special, np.ndarray):
            if special.dtype.names is None or "FORMAT" not in special.dtype.names:
                raise ValueError(
                    "Special structured array must contain a FORMAT column"
                )
            special_rows = special
            special_format = get_format(special_rows)
            if special_format != orbit_format:
                raise ValueError(
                    f"Special array FORMAT ({special_format}) does not match main array FORMAT ({orbit_format})"
                )

        fig2d_cache, fig3d_cache, special_ids = build_fig_caches(
            rows=rows,
            orbit_format=orbit_format,
            input_plane=input_plane_infer,
            input_origin=input_origin_infer,
            special_rows=special_rows,
            n_points=n_points,
            r_max=r_max,
            cache_dir=cache_dir,
        )

        run_dash_app(fig2d_cache, fig3d_cache, special_ids=special_ids)

    else:
        raise TypeError("Input data must be a file path or a numpy structured array")
