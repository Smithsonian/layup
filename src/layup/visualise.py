import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

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
    # we want an eccentrict anomaly array for the entire ellipse, but
    # to avoid all orbits starting at perihelion in the plot we will do
    # a quick numerical estimate of the eccentric anomaly from the input
    # mean anomaly via fixed point iteration and then wrap the linspace 
    # around this value from 0 to 2pi
    E_init = M        # initial guesses using mean anomaly (shouldn't be too far off)

    for tries in range(100):
        E_new = M + e * np.sin(E_init)
        if np.abs(E_new - E_init) < 1e-8:
            break
        E_init = E_new
    if tries == 99:
        print('didnt converge')
        raise ValueError("Did not converge for all M values")

    N = 2000
    start = np.linspace(E_new, 360, N//2, endpoint=True)
    end = np.linspace(0, E_new, N - len(start))
    E = np.concatenate((start, end)) * np.pi/180

    # or, if you don't care, define eccentric anomaly for all ellipses
    # E = np.linspace(0, 2*np.pi, 10000)

    # define position vectors in orthogonal reference frame with origin 
    # at ellipse main focus with q1 oriented towards periapsis (see chapter
    # 1.2 of Morbidelli 2011, "Modern Celestial Mechanics" for details)
    q1 = a * (np.cos(E) - e)
    q2 = a * np.sqrt(1. - e**2) * np.sin(E)

    if type(q1) == float:
        q = np.array([q1, q2, 0.])
    else:
        q = np.array([q1, q2, np.zeros_like(q1)])

    # define rotation matrix to map orthogonal reference frame to 
    # barycentric reference frame
    c0 = np.cos(np.pi * Omega/180)
    s0 = np.sin(np.pi * Omega/180)
    co = np.cos(np.pi * omega/180)
    so = np.sin(np.pi * omega/180)
    ci = np.cos(np.pi * i/180)
    si = np.sin(np.pi * i/180)

    R = np.array([
        [c0 * co - s0 * so * ci,  -c0 * so - s0 * co * ci,   s0 * si],
        [s0 * co + c0 * so * ci,  -s0 * so + c0 * co * ci,  -c0 * si],
        [si * so,                 si * co,                   ci     ]
    ])

    # apply rotation matrix
    r = np.einsum('ij,j...', R, q)

    return r

def plot_ellipse(
        orb_array,
        plot_type,
        output
):
    """
    Handle the plotting of the orbit ellipses depending on whether the user wants
    matplot flat 2D plots, or 3D interactive plots with plotly

    Parameters
    -----------
    orb_array : numpy structured array
        The orbits as read in by data readers
    plot_type : str
        The plotting engine to use. Can be either 'matplot' or 'plotly' currently
    output : str
        The output file stem. 
    """

    if plot_type == 'matplot':
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        plt.rcParams.update(
            {'axes.linewidth': 1.875,
            'grid.linewidth': 1.5,
            'lines.linewidth': 2.25,
            'lines.markersize': 9.0,
            'patch.linewidth': 1.5,
            'xtick.major.width': 1.875,
            'ytick.major.width': 1.875,
            'xtick.minor.width': 1.5,
            'ytick.minor.width': 1.5,
            'xtick.major.size': 9.0,
            'ytick.major.size': 9.0,
            'xtick.minor.size': 6.0,
            'ytick.minor.size': 6.0,
            'font.size': 18.0,
            'axes.labelsize': 18.0,
            'axes.titlesize': 18.0,
            'xtick.labelsize': 16.5,
            'ytick.labelsize': 16.5,
            'legend.fontsize': 16.5,
            'legend.title_fontsize': 18.0}
        )

        # TODO: make a y-z plot as well as an x-y plot and have them side-by-side?
        fig, axs = plt.subplots(figsize=(15,9))

        axs.tick_params(
            labelbottom=True,
            labeltop=True,
            labelleft=True,
            labelright=True
        )

        fig.patch.set_facecolor('k')
        axs.set_facecolor('k')

        axs.spines['bottom'].set_color('white')
        axs.spines['top'].set_color('white')
        axs.spines['right'].set_color('white')
        axs.spines['left'].set_color('white')

        axs.tick_params(axis='x', colors='white')
        axs.tick_params(axis='y', colors='white')

        axs.xaxis.label.set_color('white')
        axs.yaxis.label.set_color('white')

        axs.set_xlabel('x [AU]')
        axs.set_ylabel('y [AU]')

        for obj in orb_array:
            posvec = construct_ellipse(obj[2], obj[3], obj[4], obj[6], obj[5], obj[7])

            # doing some matplot magic to make the orbits fade in opacity further away from the final position in array
            points = np.array([posvec[:,0], posvec[:,1]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            colors = [(1,1,1,a) for a in np.power(np.linspace(0, 1, len(segments)), 3)]

            lc = LineCollection(segments, colors=colors, linewidth=1)
            axs.add_collection(lc)

        # TODO: read in semimajor values and determine max, use that to figure out plot extents?
        x_extent = 50
        y_extent = 50

        axs.set(
            xlim=(-x_extent,x_extent),
            ylim=(-y_extent,y_extent),
            aspect="equal" if x_extent == y_extent else "auto"
        )

        # TODO: redo file saving and make it go to user defined path
        plt.savefig(output, dpi=300)
        plt.show()
        plt.close()


    elif plot_type == 'plotly':
        import plotly.graph_objects as go

        fig = go.Figure()

        fig.update_layout(plot_bgcolor='rgb(0,0,0)',
                          paper_bgcolor='rgb(0,0,0)',
                          scene = dict(
                              xaxis = dict(
                              backgroundcolor="rgba(0, 0, 0,0)",
                              gridcolor="rgb(106, 110, 117)",
                              title_font=dict(color="rgb(255,255,255)"),
                              tickfont=dict(color="rgb(255,255,255)"),
                              showbackground=True,
                              zerolinecolor="white",
                            ),
                              yaxis = dict(
                              backgroundcolor="rgba(0, 0, 0,0)",
                              gridcolor="rgb(106, 110, 117)",
                              title_font=dict(color="rgb(255,255,255)"),
                              tickfont=dict(color="rgb(255,255,255)"),
                              showbackground=True,
                              zerolinecolor="white"
                            ),
                              zaxis = dict(
                              backgroundcolor="rgba(0, 0, 0,0)",
                              gridcolor="rgb(106, 110, 117)",
                              title_font=dict(color="rgb(255,255,255)"),
                              tickfont=dict(color="rgb(255,255,255)"),
                              showbackground=True, 
                              zerolinecolor="white",
                            ),
                        ),
                    )
        fig.update_yaxes(title_font_color='white', color='white')
        fig.update_xaxes(title_font_color='white', color='white')


        xs = []
        ys = []
        zs = []
        names = []

        for obj in orb_array:
            posvec = construct_ellipse(obj[2], obj[3], obj[4], obj[6], obj[5], obj[7])
            xs.append(posvec[-1,0])
            ys.append(posvec[-1,1])
            zs.append(posvec[-1,2])
            names.append(obj[0])

            normx = (posvec[:,0] - posvec[:,0].min()) / (posvec[:,0].max() - posvec[:,0].min())
            opacity = np.log1p(normx * 9) / np.log1p(9)        

            # TODO: fix hover boxes to reflect objID properly
            fig.add_trace(go.Scatter3d(
                x=posvec[:,0], y=posvec[:,1], z=posvec[:,2], 
                mode="lines+markers",
                marker=dict(color=opacity, size=5),
                line=dict(color='white'),
                customdata=[obj[0]],
                hovertemplate="<b>%{customdata[0]}</b><br>(x: %{x:.2f}, y: %{y:.2f}, z: %{z:.2f})<extra></extra>"
            ))

        # TODO: add object a/e/i as hover tooltip info?
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, 
            mode="markers",
            marker=dict(color='rgba(255, 0, 0, 1)', size=5),
            customdata=names,  # Pass planet name as custom data
            hovertemplate="<b>%{customdata}</b><br>(x: %{x:.2f}, y: %{y:.2f}, z: %{z:.2f})<extra></extra>"  # Define hover format
        ))

        fig.update_layout(showlegend=False)
        fig.show()

        # TODO: add in file saving? or can user do it from plotly output



def visualise_cli(
        input: str,
        output_file_stem: str,
        plot_type: Literal["matplot", "plotly"] = "matplot",
        input_format: Literal["BCART", "BCOM", "BKEP", "CART", "COM", "KEP"] = "BKEP",
        file_format: Literal["csv", "hdf5"] = "csv",
        num_orbs: int = 1_000,
):
    """
    Read in an orbits file and plot a user determined random number of orbits.

    Parameters
    -----------
    inputs : str
        The path to the input orbits file
    
    output : str
        The output file stem name.
    
    plot_type : str (default=matplot)
        The backend to use for plotting orbits. Must be one of: 'matplot' or 'plotly'

    input_format : str (default=BKEP)
        The orbit format of the input orbits. Must be one of: 'BCART', 'BCOM', 'BKEP', 'CART', 'COM', or 'KEP'

    file_format : str (default=csv)
        The format of the input orbits file. Must be one of: 'csv' or 'hdf5'

    num_orbs : int, optional (default=1_000)
        The number of random orbits to plot
    """
    input_file = Path(input)

    output_file = Path(f"{output_file_stem}.pdf")

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

    plot_ellipse(random_orbs, plot_type, output_file)



def main(): 

    # TODO: add in actual layup argument stuff to this, instead of hardcoded values/paths
    visualise_cli(
        input='/Users/josephmurtagh/Downloads/cent_orbs.csv',
        output_file_stem='testplot',
        plot_type='matplot',
        file_format='csv',
        num_orbs=100,
    )

if __name__ == "__main__":
    main()

