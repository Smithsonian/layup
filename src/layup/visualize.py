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

    logger.info(
        f"Creating cache of all input object and/or planet lines in heliocentric+barycentric && equatorial+ecliptic frames"
    )
    # create caches of all figure combinations
    PANELS = ("XY", "XZ", "YZ")
    fig2d_cache = {}
    fig3d_cache = {}
    for key in conic_cache.keys():
        # 3D has few variants
        fig3d_cache[key] = plotly_3D(
            lines_cache[key],
            conic_cache[key],
            sun_xyz=sunpos_cache[key],
            orbit_pos=pos_cache[key],
            planet_lines=planet_lines_cache[key],
            planet_id=planet_id,
            plot_sun=True,
            return_fig=True,
        )

        # 2D however has ref plane + origin + XY/XZ/YZ panel combinations + single or double panels
        # first do single panel variants
        for p in PANELS:
            fig2d_cache[(key[0], key[1], "single", p, None)] = plotly_2D(
                lines_cache[key],
                conic_cache[key],
                sun_xyz=sunpos_cache[key],
                orbit_pos=pos_cache[key],
                planet_lines=planet_lines_cache[key],
                planet_id=planet_id,
                plot_sun=True,
                panel=p,
                return_fig=True,
            )

        # now do double panel variants:
        for pL in PANELS:
            for pR in PANELS:
                fig2d_cache[(key[0], key[1], "double", pL, pR)] = plotly_2D(
                    lines_cache[key],
                    conic_cache[key],
                    sun_xyz=sunpos_cache[key],
                    orbit_pos=pos_cache[key],
                    planet_lines=planet_lines_cache[key],
                    planet_id=planet_id,
                    plot_sun=True,
                    panels=(pL, pR),
                    return_fig=True,
                )

    return fig2d_cache, fig3d_cache


def visualize_cli(
    input: str,
    input_plane: Optional[Literal["equatorial", "ecliptic"]] = None,
    input_origin: Optional[Literal["heliocentric", "barycentric"]] = None,
    num_orbs: int = 100,
    block_size: int = 10000,
    n_points: int = 500,
    r_max: float = 50.0,
    random: bool = False,
    cache_dir: Optional[str] = None,
):
    """
    Create visualisation plots of a given set of input orbits from the command line

    Parameters
    -----------
    input : str
        Input file path

    input_plane : str, optional (default=None)
        Input file reference plane. Must be one of "equatorial", "ecliptic"

    input_origin : str, optional (default=None)
        Input file frame of origin. Must be one of "heliocentric", "barycentric"

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

    # infer some stuff about the input
    input_origin_infer = input_origin
    input_plane_infer = input_plane
    input_format_infer = orbit_format

    # BCART_EQ can be normalised down to BCART to not break maths later,
    # and we just asign equatorial now
    if input_format_infer == "BCART_EQ":
        input_format_infer = "BCART"
        if input_plane_infer is None:
            logger.warning(
                "FORMAT=BCART_EQ implies using equatorial plane. Setting --input-plane = equatorial"
            )
            # raise Warning("FORMAT=BCART_EQ implies using equatorial plane. Setting --input-plane = equatorial")
        input_plane_infer = "equatorial"

    # infer origin if not user supplied
    if input_origin_infer is None:
        input_origin_infer = "barycentric" if input_format_infer.startswith("B") else "heliocentric"
        logger.warning(
            f"--input-origin not provided. Inferring {input_origin_infer} from input file column FORMAT={orbit_format}"
        )
        # raise Warning(f"--input-origin not provided. Inferring {input_origin_infer} from input file column FORMAT={orbit_format}")

    # infer plane if not user supplied
    if input_plane_infer is None:
        if input_format_infer in ("COM", "BCOM", "KEP", "BKEP"):
            input_plane_infer = "ecliptic"
            logger.warning(
                f"--input-plane not provided. Inferring ecliptic for input file column FORMAT={orbit_format}"
            )
            raise Warning(
                f"--input-plane not provided. Inferring ecliptic for input file column FORMAT={orbit_format}"
            )
        else:
            logger.warning(
                "--input-plane is required for CART/BCART formats unless FORMAT column encodes it (e.g. BCART_EQ)"
            )
            # raise Warning("--input-plane is required for CART/BCART formats unless FORMAT column encodes it (e.g. BCART_EQ)")

    logger.info(f"Inferred input orbit origin: {input_origin_infer}")
    logger.info(f"Inferred input orbit reference plane: {input_plane_infer}")
    logger.info(f"Inferred input orbit format: {input_format_infer}")

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

    fig2d_cache, fig3d_cache = build_fig_caches(
        rows=rows,
        orbit_format=orbit_format,
        input_plane=input_format_infer,
        input_origin=input_origin_infer,
        n_points=n_points,
        r_max=r_max,
        cache_dir=cache_dir,
    )

    logger.info(f"Running Dash web app")
    run_dash_app(fig2d_cache, fig3d_cache)


def visualize_notebook(
    data: str | Path | np.ndarray,
    input_plane: Optional[Literal["equatorial", "ecliptic"]] = None,
    input_origin: Optional[Literal["heliocentric", "barycentric"]] = None,
    num_orbs: int = 100,
    block_size: int = 10000,
    n_points: int = 500,
    r_max: float = 50.0,
    random: bool = False,
    cache_dir: Optional[str] = None,
):
    """
    Create visualisation plots of a given set of input orbits in a Jupyter notebook

    Parameters
    -----------
    input : str
        Input file path

    input_plane : str, optional (default=None)
        Input file reference plane. Must be one of "equatorial", "ecliptic"

    input_origin : str, optional (default=None)
        Input file frame of origin. Must be one of "heliocentric", "barycentric"

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
        visualize_cli(
            str(data),
            input_plane=input_plane,
            input_origin=input_origin,
            num_orbs=num_orbs,
            block_size=block_size,
            n_points=n_points,
            r_max=r_max,
            random=random,
            cache_dir=cache_dir,
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

        input_origin_infer = input_origin
        input_plane_infer = input_plane
        input_format_infer = orbit_format

        # BCART_EQ can be normalised down to BCART to not break maths later,
        # and we just asign equatorial now
        if input_format_infer == "BCART_EQ":
            input_format_infer = "BCART"
            if input_plane_infer is None:
                logger.warning(
                    "FORMAT=BCART_EQ implies using equatorial plane. Setting --input-plane = equatorial"
                )
                # raise Warning("FORMAT=BCART_EQ implies using equatorial plane. Setting --input-plane = equatorial")
            input_plane_infer = "equatorial"

        # infer origin if not user supplied
        if input_origin_infer is None:
            input_origin_infer = "barycentric" if input_format_infer.startswith("B") else "heliocentric"
            logger.warning(
                f"--input-origin not provided. Inferring {input_origin_infer} from input file format column FORMAT={orbit_format}"
            )
            # raise Warning(f"--input-origin not provided. Inferring {input_origin_infer} from input file column FORMAT={orbit_format}")

        # infer plane if not user supplied
        if input_plane_infer is None:
            if input_format_infer in ("COM", "BCOM", "KEP", "BKEP"):
                input_plane_infer = "ecliptic"
                logger.warning(
                    f"--input-plane not provided. Inferring ecliptic for input file format column FORMAT={orbit_format}"
                )
                # raise Warning(f"--input-plane not provided. Inferring ecliptic for input file column FORMAT={orbit_format}")
            else:
                logger.warning(
                    "--input-plane is required for CART/BCART formats unless FORMAT column encodes it (e.g. BCART_EQ)"
                )
                # raise Warning("--input-plane is required for CART/BCART formats unless FORMAT column encodes it (e.g. BCART_EQ)")

        logger.info(f"Inferred input orbit origin: {input_origin_infer}")
        logger.info(f"Inferred input orbit reference plane: {input_plane_infer}")
        logger.info(f"Inferred input orbit format: {input_format_infer}")

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

        fig2d_cache, fig3d_cache = build_fig_caches(
            rows=rows,
            orbit_format=orbit_format,
            input_plane=input_format_infer,
            input_origin=input_origin_infer,
            n_points=n_points,
            r_max=r_max,
            cache_dir=cache_dir,
        )

        run_dash_app(fig2d_cache, fig3d_cache)

    else:
        raise TypeError("Input data must be a file path or a numpy structured array")
