import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
import matplotlib as mpl

from layup.utilities.file_io.CSVReader import CSVDataReader
from layup.utilities.file_io.HDF5Reader import HDF5DataReader

logger = logging.getLogger(__name__)

# Required column names for each orbit format
REQUIRED_COLUMN_NAMES = {
    "BCART": ["ObjID", "FORMAT", "x", "y", "z", "xdot", "ydot", "zdot", "epochMJD_TDB"],
    "BCOM": ["ObjID", "FORMAT", "q", "e", "inc", "node", "argPeri", "t_p_MJD_TDB", "epochMJD_TDB"],
    "BKEP": ["ObjID", "FORMAT", "a", "e", "inc", "node", "argPeri", "ma", "epochMJD_TDB"],
    "CART": ["ObjID", "FORMAT", "x", "y", "z", "xdot", "ydot", "zdot", "epochMJD_TDB"],
    "COM": ["ObjID", "FORMAT", "q", "e", "inc", "node", "argPeri", "t_p_MJD_TDB", "epochMJD_TDB"],
    "KEP": ["ObjID", "FORMAT", "a", "e", "inc", "node", "argPeri", "ma", "epochMJD_TDB"],
}


def construct_ellipse(
    a: float,
    e: float,
    i: float,
    omega: float,
    Omega: float,
    M: float,
):
    """
    Construct an ellipse of position vectors using the formalism defined
    in chapter 1.2 of Morbidelli 2011, "Modern Celestial Mechanics"

    Paramters
    ----------
    a : float
        The semimajor axis of the orbit (in au)

    e : float
        The eccentricty of the orbit

    i : float
        The inclination of the orbit (in degrees)

    omega : float
        The argument of perihelion of the orbit (in degrees)

    Omega : float
        The longitude of the ascending node of the orbit (in degrees)

    M : float
        The mean anomaly of the orbit (in degrees)

    Returns
    --------
    r : numpy array
        The rotated barycentric position vectors of the orbit ellipse
    """
    # we want an eccentric anomaly array for the entire ellipse, but
    # to avoid all orbits starting at perihelion in the plot we will do
    # a quick numerical estimate of the eccentric anomaly from the input
    # mean anomaly via fixed point iteration and then wrap the linspace
    # around this value from 0 to 2pi
    E_init = M  # initial guesses using mean anomaly (shouldn't be too far off)

    for tries in range(100):
        E_new = M + e * np.sin(E_init)
        if np.abs(E_new - E_init) < 1e-8:
            break
        E_init = E_new
    if tries == 99:
        print("didnt converge")
        raise ValueError("Did not converge for all M values")

    N = 100
    start = np.linspace(E_new, 360, N // 2, endpoint=True)
    end = np.linspace(0, E_new, N - len(start))
    E = np.concatenate((start, end)) * np.pi / 180

    # or, if you don't care, define eccentric anomaly for all ellipses
    # E = np.linspace(0, 2*np.pi, 10000)

    # define position vectors in orthogonal reference frame with origin
    # at ellipse main focus with q1 oriented towards periapsis (see chapter
    # 1.2 of Morbidelli 2011, "Modern Celestial Mechanics" for details)
    q1 = a * (np.cos(E) - e)
    q2 = a * np.sqrt(1.0 - e**2) * np.sin(E)

    if type(q1) == float:
        q = np.array([q1, q2, 0.0])
    else:
        q = np.array([q1, q2, np.zeros_like(q1)])

    # define rotation matrix to map orthogonal reference frame to
    # barycentric reference frame
    c0 = np.cos(np.pi * Omega / 180)
    s0 = np.sin(np.pi * Omega / 180)
    co = np.cos(np.pi * omega / 180)
    so = np.sin(np.pi * omega / 180)
    ci = np.cos(np.pi * i / 180)
    si = np.sin(np.pi * i / 180)

    R = np.array(
        [
            [c0 * co - s0 * so * ci, -c0 * so - s0 * co * ci, s0 * si],
            [s0 * co + c0 * so * ci, -s0 * so + c0 * co * ci, -c0 * si],
            [si * so, si * co, ci],
        ]
    )

    # apply rotation matrix
    r = np.einsum("ij,j...", R, q)

    return r


@mpl.rc_context(
    {
        "axes.linewidth": 1.875,
        "grid.linewidth": 1.5,
        "lines.linewidth": 2.25,
        "lines.markersize": 9.0,
        "patch.linewidth": 1.5,
        "xtick.major.width": 1.875,
        "ytick.major.width": 1.875,
        "xtick.minor.width": 1.5,
        "ytick.minor.width": 1.5,
        "xtick.major.size": 9.0,
        "ytick.major.size": 9.0,
        "xtick.minor.size": 6.0,
        "ytick.minor.size": 6.0,
        "font.size": 18.0,
        "axes.labelsize": 18.0,
        "xtick.labelsize": 16.5,
        "ytick.labelsize": 16.5,
        "legend.fontsize": 16.5,
        "legend.title_fontsize": 18.0,
        "axes.titlesize": 32,
    }
)
def matplot_2D(orb_array, planets, no_planets, no_sun, output, fade):
    """
    Create a 2D orbit distribution plot using matplot as a backend.

    Parameters
    -------
    orb_array : structured numpy array
        Numpy array containing orbital element information

    planets : list
        List of desired planet orbits to be plotted

    no_planets : bool
        Flag to turn off planet orbits

    no_sun : bool
        Flag to turn of sun plotting

    output : str
        Output file stem

    fade : bool
        Flag to turn on object orbit fading
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, axs = plt.subplots(1, 2, layout="constrained", figsize=(20, 9))
    fig.patch.set_facecolor("k")

    for ax in axs:
        ax.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True, colors="white")

        ax.set_facecolor("k")

        for spine in ax.spines.values():
            spine.set_color("white")

        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

        if np.max(orb_array["a"] < 2):
            maxQ = np.max(orb_array["a"] * (1 + orb_array["e"])) + 0.2
            ax.set(xlim=(-maxQ, maxQ), ylim=(-maxQ, maxQ), aspect="equal")
        elif np.max(orb_array["a"] <= 5):
            maxQ = np.max(orb_array["a"] * (1 + orb_array["e"])) + 1
            ax.set(xlim=(-maxQ, maxQ), ylim=(-maxQ, maxQ), aspect="equal")
        else:
            maxQ = np.max(orb_array["a"] * (1 + orb_array["e"])) + 5
            ax.set(xlim=(-maxQ, maxQ), ylim=(-maxQ, maxQ), aspect="equal")

    # add objects
    for obj in orb_array:
        posvec = construct_ellipse(obj[2], obj[3], obj[4], obj[6], obj[5], obj[7])

        # doing some matplot magic to make the orbits fade in opacity further away from the final position in array
        points = np.array([posvec[:, 0], posvec[:, 1]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        if fade:
            colors = [
                (14 / 255, 84 / 255, 118 / 255, a) for a in np.power(np.linspace(0, 1, len(segments)), 5)
            ]
            lc = LineCollection(segments, colors=colors, linewidth=1)
        else:
            lc = LineCollection(segments, colors=(14 / 225, 84 / 225, 118 / 225, 0.6), linewidth=1)
        axs[0].add_collection(lc)

        points = np.array([posvec[:, 0], posvec[:, 2]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        if fade:
            colors = [
                (14 / 255, 84 / 255, 118 / 255, a) for a in np.power(np.linspace(0, 1, len(segments)), 5)
            ]
            lc = LineCollection(segments, colors=colors, linewidth=1, zorder=10)
        else:
            lc = LineCollection(segments, colors=(14 / 225, 84 / 225, 118 / 225, 0.6), linewidth=1)
        axs[1].add_collection(lc)

    # add the sun
    if not no_sun:
        axs[0].scatter(0, 0, s=100, color="xkcd:sunflower yellow")

    # add planets
    if not no_planets:
        planets_dic = {
            "Me": [0.387, 7.0, "#E7E8EC"],
            "V": [0.723, 3.4, "#E39E1C"],
            "E": [1.0, 0, "#6B93D6"],
            "Ma": [1.524, 1.9, "#C1440E"],
            "J": [5.204, 1.3, "#D8CA9D"],
            "S": [9.582, 2.5, "#EAD6B8"],
            "U": [19.218, 0.8, "#D1E7E7"],
            "N": [30.11, 1.8, "#85ADDB"],
        }

        theta = np.linspace(0, 2 * np.pi, 200)
        dictfilt = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])

        planets_dic = dictfilt(planets_dic, planets)
        for _, (k, v) in enumerate(planets_dic.items()):
            axs[0].add_patch(plt.Circle((0, 0), v[0], color=v[2], fill=False, alpha=0.9, linestyle="dotted"))
            # axs[0].text(v[0]+1, 0, k[0], color=v[2])

            axs[1].plot(
                v[0] * np.cos(theta),
                v[0] * np.cos(theta) * np.sin(np.radians(v[1])),
                color=v[2],
                linestyle="dotted",
            )

    axs[0].set_xlabel("x [AU]")
    axs[0].set_ylabel("y [AU]")
    axs[1].set_xlabel("x [AU]")
    axs[1].set_ylabel("z [AU]")

    axs[0].set_title("Bird's Eye", fontdict={"color": "white"})
    axs[1].set_title("Edge-on", fontdict={"color": "white"})

    plt.savefig(output, dpi=300)
    plt.show()
    plt.close()


def matplot_3D(orb_array, planets, no_planets, no_sun, output, fade):
    """
    Create a 3D orbit distribution plot using matplot as a backend.

    Parameters
    -------
    orb_array : structured numpy array
        Numpy array containing orbital element information

    planets : list
        List of desired planet orbits to be plotted

    no_planets : bool
        Flag to turn off planet orbits

    no_sun : bool
        Flag to turn of sun plotting

    output : str
        Output file stem

    fade : bool
        Flag to turn on object orbit fading
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig = plt.figure(figsize=(15, 9))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axs = fig.add_subplot(111, projection="3d")

    fig.patch.set_facecolor("black")
    axs.set_facecolor("black")
    axs.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))  # RGBA: white
    axs.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    axs.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))

    axs.xaxis._axinfo["grid"]["color"] = (0.1, 0.1, 0.1, 1)  # grey
    axs.yaxis._axinfo["grid"]["color"] = (0.1, 0.1, 0.1, 1)  # grey
    axs.zaxis._axinfo["grid"]["color"] = (0.1, 0.1, 0.1, 1)  # grey

    axs.xaxis.label.set_color("white")
    axs.yaxis.label.set_color("white")
    axs.zaxis.label.set_color("white")

    axs.tick_params(axis="x", colors="white", length=0)
    axs.tick_params(axis="y", colors="white", length=0)
    axs.tick_params(axis="z", colors="white", length=0)

    for axis in [axs.xaxis, axs.yaxis, axs.zaxis]:
        axis._axinfo["tick"]["inward_factor"] = 0
        axis._axinfo["tick"]["outward_factor"] = 0
        axis._axinfo["tick"]["size"] = 0

    axs.set_xlabel("x [au]")
    axs.set_ylabel("y [au]")
    axs.set_zlabel("z [au]")

    for obj in orb_array:
        posvec = construct_ellipse(obj[2], obj[3], obj[4], obj[6], obj[5], obj[7])

        points = np.array([posvec[:, 0], posvec[:, 1], posvec[:, 2]]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        if fade:
            alpha = np.power(np.linspace(0, 1, len(segments)), 3)
            colors = np.column_stack(
                (np.ones(len(segments)), np.ones(len(segments)), np.ones(len(segments)), alpha)  # R  # G  # B
            )
            lc = Line3DCollection(segments, colors=colors, linewidth=1)
        else:
            lc = Line3DCollection(segments, colors=(14 / 225, 84 / 225, 118 / 225, 0.6), linewidth=1)
        axs.add_collection3d(lc)

    # add sun
    if not no_sun:
        axs.scatter(0, 0, s=100, color="xkcd:sunflower yellow")

    # add planets
    if not no_planets:
        planets_dic = {
            "Me": [0.387, 7.0, "#E7E8EC"],
            "V": [0.723, 3.4, "#E39E1C"],
            "E": [1.0, 0, "#6B93D6"],
            "Ma": [1.524, 1.9, "#C1440E"],
            "J": [5.204, 1.3, "#D8CA9D"],
            "S": [9.582, 2.5, "#EAD6B8"],
            "U": [19.218, 0.8, "#D1E7E7"],
            "N": [30.11, 1.8, "#85ADDB"],
        }

        theta = np.linspace(0, 2 * np.pi, 200)
        dictfilt = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])

        planets_dic = dictfilt(planets_dic, planets)
        for _, (k, v) in enumerate(planets_dic.items()):
            axs.plot(
                v[0] * np.cos(theta),
                v[0] * np.sin(theta) * np.cos(np.radians(v[1])),
                v[0] * np.cos(theta) * np.sin(np.radians(v[1])),
                color=v[2],
                alpha=0.9,
                linestyle="dotted",
            )
            axs.text(v[0], 0, 0, k[0], color=v[2])

    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.show()


def plotly_3D(orb_array, planets, no_planets, no_sun, output, fade):
    """
    Create a 3D orbit distribution plot using plotly as a backend.

    Parameters
    -------
    orb_array : structured numpy array
        Numpy array containing orbital element information

    planets : list
        List of desired planet orbits to be plotted

    no_planets : bool
        Flag to turn off planet orbits

    no_sun : bool
        Flag to turn of sun plotting

    output : str
        Output file stem

    fade : bool
        Flag to turn on object orbit fading
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.update_layout(
        plot_bgcolor="rgb(0,0,0)",
        paper_bgcolor="rgb(0,0,0)",
        scene=dict(
            xaxis=dict(
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="rgb(106, 110, 117)",
                title_font=dict(color="rgb(255,255,255)"),
                tickfont=dict(color="rgb(255,255,255)"),
                showbackground=True,
                zerolinecolor="white",
            ),
            yaxis=dict(
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="rgb(106, 110, 117)",
                title_font=dict(color="rgb(255,255,255)"),
                tickfont=dict(color="rgb(255,255,255)"),
                showbackground=True,
                zerolinecolor="white",
            ),
            zaxis=dict(
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="rgb(106, 110, 117)",
                title_font=dict(color="rgb(255,255,255)"),
                tickfont=dict(color="rgb(255,255,255)"),
                showbackground=True,
                zerolinecolor="white",
            ),
        ),
    )

    fig.update_yaxes(title_font_color="white", color="white")
    fig.update_xaxes(title_font_color="white", color="white")

    if not no_sun:
        fig.add_trace(go.scatter(x=0, y=0, s=100, color="xkcd:sunflower yellow"))

    for obj in orb_array:
        posvec = construct_ellipse(obj[2], obj[3], obj[4], obj[6], obj[5], obj[7])

        normx = (posvec[:, 0] - posvec[:, 0].min()) / (posvec[:, 0].max() - posvec[:, 0].min())
        opacity = np.log1p(normx * 9) / np.log1p(9)

        fig.add_trace(
            go.Scatter(
                x=posvec[:, 0],
                y=posvec[:, 1],
                mode="lines",
                line=dict(
                    color=opacity,
                    colorscale="gray",
                    cmin=0,
                    cmax=1,
                ),
                customdata=[obj[0]],
                hovertemplate="<b>%{customdata[0]}</b><br>(x: %{x:.2f}, y: %{y:.2f}, z: %{z:.2f})<extra></extra>",
            )
        )

    fig.update_layout(showlegend=False)
    fig.show()


def visualize_cli(
    input: str,
    output_file_stem: str,
    planets: list,
    file_format: Literal["csv", "hdf5"] = "csv",
    input_format: Literal["BCART", "BCOM", "BKEP", "CART", "COM", "KEP"] = "BKEP",
    backend: Literal["matplot", "plotly"] = "matplot",
    dimensions: Literal["2D", "3D"] = "2D",
    num_orbs: int = 1_000,
    no_planets: bool = False,
    no_sun: bool = False,
    fade: bool = False,
):
    """
    Read in an orbits file and plot a user determined random number of orbits.

    Parameters
    -----------
    inputs : str
        The path to the input orbits file

    output : str
        The output file stem name.

    planets : list
        List of planets to be plotted from ['Me', 'V', 'E', 'Ma', J', 'S', 'U', 'N]

    file_format : str (default=csv)
        The format of the input orbits file. Must be one of: 'csv' or 'hdf5'

    input_format : str (default=BKEP)
        The orbit format of the input orbits. Must be one of: 'BCART', 'BCOM', 'BKEP', 'CART', 'COM', or 'KEP'

    backend : str (default=matplot)
        The backend to use for plotting orbits. Must be one of: 'matplot' or 'plotly'

    dimensions : str (default=2D)
        Number of dimensions for plotting purposes. Must be one of: '2D' or '3D' (TODO: add 4D plotting ;))

    num_orbs : int (default=1_000)
        The number of random orbits to plot

    no_planets : bool (default=False)
        Flag to turn off the planet orbits

    no_sun : bool (default=False)
        Flag to turn off the Sun

    fade : bool (default=False)
        Flag to turn off object orbit fading
    """
    input_file = Path(input)

    output_file = Path(f"{output_file_stem}")

    output_directory = output_file.parent.resolve()
    if not output_directory.exists():
        logging.error(f"Output directory {output_directory} does not exist")

    # TODO: currently assumes BKEP input. need to convert if not in BKEP
    required_columns = REQUIRED_COLUMN_NAMES[input_format]

    if file_format == "hdf5":
        reader = HDF5DataReader(
            input_file,
            form_column_name="FORMAT",
            required_column_names=required_columns,
        )
    else:
        reader = CSVDataReader(
            input_file,
            format_column_name="FORMAT",
            required_column_names=required_columns,
        )

    random_orbs = reader.read_rows(block_start=0, block_size=1000)
    random_orbs = random_orbs[np.random.choice(random_orbs.size, size=num_orbs, replace=False)]

    if backend == "matplot":
        if dimensions == "2D":
            matplot_2D(random_orbs, planets, no_planets, no_sun, output_file_stem, fade)
        else:
            matplot_3D(random_orbs, planets, no_planets, no_sun, output_file_stem, fade)
    elif backend == "plotly":
        if dimensions == "3D":
            plotly_3D(random_orbs)
