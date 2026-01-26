import logging
from typing import Literal, Tuple, Optional, Any

import numpy as np
from datetime import datetime, timedelta

from layup.orbit_maths import ClassicalConic

PANEL = Literal["XY", "XZ", "YZ"]

PLANET_COLOURS_NIGHT = {
    "Mercury": "rgba(190,190,190,0.95)",  # silver
    "Venus": "rgba(255,190,90,0.95)",  # warm amber
    "Earth": "rgba(90,210,255,0.95)",  # cyan
    "Mars": "rgba(255,90,90,0.95)",  # red
    "Jupiter": "rgba(255,165,120,0.95)",  # salmon/orange
    "Saturn": "rgba(255,230,150,0.95)",  # pale gold
    "Uranus": "rgba(120,255,210,0.95)",  # mint
    "Neptune": "rgba(185,120,255,0.95)",  # purple
}

PLANET_COLOURS_DAY = {
    "Mercury": "rgba(90,90,90,0.95)",  # dark gray
    "Venus": "rgba(180,110,0,0.95)",  # brown/orange
    "Earth": "rgba(0,140,190,0.95)",  # teal-blue
    "Mars": "rgba(170,0,0,0.95)",  # dark red
    "Jupiter": "rgba(170,90,60,0.95)",  # brown
    "Saturn": "rgba(160,140,40,0.95)",  # olive gold
    "Uranus": "rgba(0,150,110,0.95)",  # green-teal
    "Neptune": "rgba(115,0,170,0.95)",  # deep purple
}

logger = logging.getLogger(__name__)


# --- set up plots ---
def add_reference_plane_xy(fig, lines: np.ndarray, planet_lines: np.ndarray, opacity: float = 0.10):
    """
    Add the reference plane (either ecliptic or equatorial) to the 3D plot at Z=0

    Parameters
    -----------
    fig : object
        Plotly figure object

    lines : numpy float array
        Array of x,y,z coordinates of objects to be plotted

    lines : numpy float array
        Array of x,y,z coordinates of planets to be plotted

    opacity : float, optional (default = 0.10)
        Transparency of the reference plane
    """
    import plotly.graph_objects as go

    # extract all x/y coords and flatten (ravel) to 1D
    x_obj = lines[..., 0].ravel()
    y_obj = lines[..., 1].ravel()

    # either take the extent of the plane to be the furthest orbit or furthest planet
    if planet_lines is not None and np.size(planet_lines) > 0:
        x_pla = planet_lines[..., 0].ravel()
        y_pla = planet_lines[..., 1].ravel()
        x0 = min(np.nanmin(x_obj), np.nanmin(x_pla))
        x1 = max(np.nanmax(x_obj), np.nanmax(x_pla))
        y0 = min(np.nanmin(y_obj), np.nanmin(y_pla))
        y1 = max(np.nanmax(y_obj), np.nanmax(y_pla))
    else:
        x0, x1 = np.nanmin(x_obj), np.nanmax(x_obj)
        y0, y1 = np.nanmin(y_obj), np.nanmax(y_obj)

    # add a bit of padding from the edges
    padx = 0.05 * (x1 - x0) if (x1 - x0) > 0 else 1.0
    pady = 0.05 * (y1 - y0) if (y1 - y0) > 0 else 1.0
    x0, x1 = x0 - padx, x1 + padx
    y0, y1 = y0 - pady, y1 + pady

    # now create the plane as nx x ny grid
    nx, ny = 40, 40
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)

    # add to the figure
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            showscale=False,
            opacity=opacity,
            surfacecolor=np.zeros_like(X),
            hoverinfo="skip",
            name="ref-plane",
            hovertemplate=None,
        )
    )


# --- very crude orbit classifier ---
def classify(a: float, e: float, Tj: float) -> str:
    """
    Rough first pass orbit classifier based on input orbital elements. Does NOT do any
    orbit integrations to verify e.g. Trojan or other resonant behaviour, so use only as
    an approximate guess. Check your orbits thoroughly.

    Parameters
    -----------
    a : float
        Semimajor axis of the object, in units of AU

    e : float
        Eccentricity of the object

    Tj : float
        Tisserand parameter with respect to Jupiter of the object
    """
    q = a * (1 - e)

    # these are some very very crude dynamical classifiers for filtering
    # purposes, don't take them as gospel - do your own checks!!
    if e >= 1.0:
        return "Hyperbolic"
    elif np.isfinite(a) and a > 2000:
        return "Inner Oort Cloud"
    elif np.isfinite(Tj) and (Tj < 2.0):
        return "LPC"
    elif np.isfinite(Tj) and (2.0 <= Tj < 3.0) and (q < 5.2):
        return "JFC"
    elif np.isfinite(q) and (q < 1.3):
        return "NEO"
    elif np.isfinite(q) and (1.3 <= q < 1.66):
        return "Mars-Crosser"
    elif np.isfinite(a) and (2.0 <= a <= 3.5) and np.isfinite(q) and (q >= 1.66):
        return "MBA"
    elif np.isfinite(a) and (a < 30.04) and np.isfinite(q) and (q >= 7.35):
        return "Centaur"
    elif np.isfinite(a) and (30.04 <= a < 2000.0) and np.isfinite(q) and (q >= 7.35):
        return "TNO"
    else:
        return "Other"


# --- plot in 2D ---
def plotly_2D(
    lines: np.ndarray,
    canon: ClassicalConic,
    plot_sun: bool = True,
    orbit_pos: Optional[np.ndarray] = None,
    sun_xyz: Optional[np.ndarray] = None,
    planet_lines: Optional[np.ndarray] = None,
    planet_id: Optional[np.ndarray] = None,
    return_fig: bool = False,
    output: Optional[str] = None,
    panel: Optional[PANEL] = None,
    panels: Optional[Tuple[PANEL, PANEL]] = None,
):
    """
    Create a 2D (1x2 subplot) interactive Plotly figure of orbits

    Parameters
    -----------
    lines : dict of numpy array
        Dictionary of arrays with the orbit lines for each object in each plane+origin combination

    canon : ClassicalConic object
        Object with the conic section class instances of each object and their properties

    plot_sun : bool, optional (default = True)
        Flag to turn on/off plotting the Sun

    orbit_pos : dict of numpy array, optional (default = None)
        Dictionary of arrays with object positions in each plane+origin combination

    sun_xyz : dict of numpy array, optional (default = None)
        Dictionary of arrays with the Sun positions in each plane+origin combination

    planet_lines : dict of arrays, optional (default = None)
        Dictionary of arrays containing planet orbit lines for each planet in each plane+origin
        combination of shape (n_planets, n_points, 3)

    planet_id : numpy string array, optional (default = None)
        Array containing ID tags for each planet of shape (n_planets,)

    return_fig : bool, optional (default = False)
        Flag to turn on/off returning the figure object

    output : str, optional (default = None)
        String containing the html of the figure

    panel: str, optional (default = None)
        String containing which orientation to draw a single panel of. Must be one of "XY", "XZ", "YZ"

    panels: str, optional (default = None)
        String containing which orientation to draw two panels of. Must be one of "XY", "XZ", "YZ"

    Returns
    --------
    fig : object
        Plotly figure object
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def coords_for(p: PANEL, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        if p == "XY":
            return (
                x,
                y,
                "X [AU]",
                "Y [AU]",
                True,
            )  # <-- the final True here is letting later on know to plot this panel equal aspect, as top-down is onto the reference plane
        if p == "XZ":
            return (
                x,
                z,
                "X [AU]",
                "Z [AU]",
                False,
            )  # <-- this and YZ are inclination driven so shouldn't be equal aspect
        if p == "YZ":
            return y, z, "Y [AU]", "Z [AU]", False
        logger.error(f"Unknown panel {p!r} (expected 'XY', 'XZ', 'YZ')")
        raise ValueError(f"Unknown panel {p!r} (expected 'XY', 'XZ', 'YZ')")

    # -- panel configuration --
    if panels is not None:
        panels_to_show: Tuple[PANEL, ...] = tuple(
            panels
        )  # < -- if >1 panels, turn whichever options they are into tuple
    elif panel is not None:
        panels_to_show = (panel,)  # < -- if 1 panel, turn that option into tuple
    else:
        panels_to_show = ("XY", "XZ")  # < -- if none specified, default to 2 panel XY+XZ

    if len(panels_to_show) not in (1, 2):
        logger.error(f"Expected 1 or 2 panels, got {len(panels_to_show)}: {panels_to_show}")
        raise ValueError(f"Expected 1 or 2 panels, got {len(panels_to_show)}: {panels_to_show}")

    ncols = len(panels_to_show)

    fig = make_subplots(rows=1, cols=ncols, horizontal_spacing=0.10 if ncols == 2 else 0.02)

    # -- plot planets --
    if planet_lines is not None:
        if planet_id is None:
            # in the case you have planet lines but no id, just assign them generic name tags
            planet_id = np.array([f"Planet {i}" for i in range(planet_lines.shape[0])], dtype="U32")

        for i in range(planet_lines.shape[0]):
            # get planet coords and colour by searching the global variable (default is night mode,
            # we can change that in the dash app and css later)
            x = planet_lines[i, :, 0]
            y = planet_lines[i, :, 1]
            z = planet_lines[i, :, 2]
            colour = PLANET_COLOURS_NIGHT.get(str(planet_id[i]), "rgba(200,200,200,0.6)")

            hover_text = f"{planet_id[i]}<extra></extra>"

            # loop over however many panels we have
            for col, p in enumerate(panels_to_show, start=1):
                xa, ya, _, _, _ = coords_for(
                    p, x, y, z
                )  # <-- grab the correct axes x/y coords for whatever plane it is
                fig.add_trace(
                    go.Scatter(
                        x=xa,
                        y=ya,
                        mode="lines",
                        line=dict(color=colour, width=2.2),
                        hovertemplate=hover_text,
                        showlegend=False,
                        name=str(planet_id[i]),
                        meta={"kind": "Planet"},  # <-- this tag is to prevent colours being overwritten later
                    ),
                    row=1,
                    col=col,
                )

    # -- plot input objects --
    for i in range(lines.shape[0]):
        # get object coords
        x = lines[i, :, 0]
        y = lines[i, :, 1]
        z = lines[i, :, 2]

        # it may not be best practice to paste unicode symbols here but ¯\_(ツ)_/¯
        hover_text = (
            f"{canon.obj_id[i]}<br>"
            f"e: {canon.e[i]:.4f}<br>"
            f"i: {np.rad2deg(canon.inc[i]):.2f}°<br>"
            f"Ω: {np.rad2deg(canon.node[i]):.2f}°<br>"
            f"ω: {np.rad2deg(canon.argp[i]):.2f}°"
        )

        # these are being extracted so we can crudely classify them later
        L = float(canon.L[i])
        e = float(canon.e[i])
        inc = float(canon.inc[i])
        if abs(1 - e**2) < 1e-12:  # <-- protect against parabolic orbits
            a = np.inf
        else:
            a = L / (1 - e**2)
        if (not np.isfinite(a)) or (a == 0.0):  # <-- same again
            Tj = np.nan
        else:
            Tj = (5.2044 / a) + 2.0 * np.cos(inc) * np.sqrt((a / 5.2044) * (1 - e**2))
        pop = classify(a, e, Tj)

        # loop over however many panels we have
        for col, p in enumerate(panels_to_show, start=1):
            xa, ya, _, _, _ = coords_for(
                p, x, y, z
            )  # <-- grab the correct axes x/y coords for whatever plane it is
            fig.add_trace(
                go.Scatter(
                    x=xa,
                    y=ya,
                    mode="lines",
                    line=dict(color="rgba(144,167,209,0.7)", width=1.5),
                    hovertemplate=hover_text + "<extra></extra>",
                    showlegend=False,
                    name=str(canon.obj_id[i]),
                    meta={"kind": pop},
                ),
                row=1,
                col=col,
            )

    # -- plot input object epoch position --
    if orbit_pos is not None:
        # want to get a hover label with epoch info so first
        # extract from conic objects
        mjd = np.asarray(canon.epochMJD_TDB, dtype=float)
        mjd_str = np.array([f"{m:.5f}" for m in mjd], dtype="U32")

        # also good to have it in YYYY MM DD format
        ymd_str = np.empty(mjd.shape[0], dtype="U10")
        base = datetime(1858, 11, 17)  # <-- MJD 0
        for i, mjd in enumerate(mjd):
            dt = base + timedelta(days=float(mjd))
            ymd_str[i] = dt.strftime("%Y %b %d")

        # vectorise it by stacking all labels together to add
        # as one trace into plotly
        mjd_stack = np.column_stack([mjd_str, ymd_str])
        hover_text = "%{text}<br>" "@ MJD %{customdata[0]}<br>" "(%{customdata[1]})" "<extra></extra>"

        # loop over however many panels we have
        for col, p in enumerate(panels_to_show, start=1):
            xa, ya, _, _, _ = coords_for(
                p, orbit_pos[:, 0], orbit_pos[:, 1], orbit_pos[:, 2]
            )  # <-- grab the correct axes x/y coords for whatever plane it is
            fig.add_trace(
                go.Scatter(
                    x=xa,
                    y=ya,
                    mode="markers",
                    marker=dict(size=5, color="rgba(255,255,255,0.9)"),
                    showlegend=False,
                    text=np.asarray(canon.obj_id),
                    customdata=mjd_stack,
                    hovertemplate=hover_text,
                    meta={"kind": "epoch"},
                ),
                row=1,
                col=col,
            )

    # -- plot the sun --
    if plot_sun:
        if sun_xyz is None:
            sx, sy, sz = 0.0, 0.0, 0.0
        else:
            sx, sy, sz = float(sun_xyz[0]), float(sun_xyz[1]), float(sun_xyz[2])

        # loop over however many panels we have
        for col, p in enumerate(panels_to_show, start=1):
            xa, ya, _, _, _ = coords_for(
                p, np.array([sx]), np.array([sy]), np.array([sz])
            )  # <-- grab the correct axes x/y coords for whatever plane it is
            fig.add_trace(
                go.Scatter(
                    x=xa,
                    y=ya,
                    mode="markers",
                    marker=dict(size=10, color="yellow"),
                    showlegend=False,
                    name="Sun",
                ),
                row=1,
                col=col,
            )

    # -- pretty up the figure --
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        autosize=True,
        margin=dict(l=60, r=60, t=40, b=60),
        hoverdistance=0,
        hovermode="closest",
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")

    # axis titles + equal aspect for any XY panel
    # loop over however many panels we have
    for col, p in enumerate(panels_to_show, start=1):
        _, _, xtitle, ytitle, want_equal = coords_for(
            p, np.array([0.0]), np.array([0.0]), np.array([0.0])
        )  # <-- grab the correct axes x/y titles for whatever plane it is
        fig.update_xaxes(title_text=xtitle, row=1, col=col)
        fig.update_yaxes(title_text=ytitle, row=1, col=col)
        if want_equal:
            if col == 1:
                fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
            elif col == 2:
                fig.update_layout(yaxis2=dict(scaleanchor="x2", scaleratio=1))

    # return figure object if user wants
    if return_fig:
        return fig

    # otherwise write out and show
    if output:
        fig.write_html(output)
        fig.show()

    fig.show()


# --- plot in 3D ---
def plotly_3D(
    lines: np.ndarray,
    canon: ClassicalConic,
    plot_sun: bool = True,
    show_plane: bool = True,
    orbit_pos: Optional[np.ndarray] = None,
    planet_lines: Optional[np.ndarray] = None,
    planet_id: Optional[np.ndarray] = None,
    sun_xyz: Optional[np.ndarray] = None,
    return_fig: bool = False,
    output: Optional[str] = None,
):
    """
    Create a 3D interactive Plotly figure of orbits

    Parameters
    -----------
    lines : dict of numpy array
        Dictionary of arrays with the orbit lines for each object in each plane+origin combination

    canon : dict of objects
        Dictionary with the conic section class instances of each object and their properties

    plot_sun : bool, optional (default = True)
        Flag to turn on/off plotting the Sun

    show_plane : bool, optional (default = True)
        Flag to turn on/off plotting the reference plane (ecliptic or equatorial)

    orbit_pos : dict of numpy array, optional (default = None)
        Dictionary of arrays with object positions in each plane+origin combination

    planet_lines : dict of arrays, optional (default = None)
        Dictionary of arrays containing planet orbit lines for each planet in each plane+origin
        combination of shape (n_planets, n_points, 3)

    planet_id : numpy string array, optional (default = None)
        Array containing ID tags for each planet of shape (n_planets,)

    sun_xyz : dict of numpy array, optional (default = None)
        Dictionary of arrays with the Sun positions in each plane+origin combination

    return_fig : bool, optional (default = False)
        Flag to turn on/off returning the figure object

    output : str, optional (default = None)
        String containing the html of the figure

    Returns
    --------
    fig : object
        Plotly figure object
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # -- plot reference plane --
    if show_plane:
        add_reference_plane_xy(fig, lines, planet_lines, opacity=0.50)

    # -- plot planets --
    if planet_lines is not None:
        if planet_id is None:
            planet_id = np.array([f"Planet {i}" for i in range(planet_lines.shape[0])], dtype="U32")

        for i in range(planet_lines.shape[0]):
            # get planet coords and colour by searching the global variable (default is night mode,
            # we can change that in the dash app and css later)
            x = planet_lines[i, :, 0]
            y = planet_lines[i, :, 1]
            z = planet_lines[i, :, 2]
            colour = PLANET_COLOURS_NIGHT.get(str(planet_id[i]), "rgba(200,200,200,0.6)")

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    line=dict(color=colour, width=5),
                    hovertemplate=f"{planet_id[i]}<extra></extra>",
                    showlegend=False,
                    name=str(planet_id[i]),
                    meta={"kind": "Planet"},  # <-- this tag is to prevent colours being overwritten later
                )
            )

    # -- plot input objects --
    for i in range(lines.shape[0]):
        x = lines[i, :, 0]
        y = lines[i, :, 1]
        z = lines[i, :, 2]

        # it may not be best practice to paste unicode symbols here but ¯\_(ツ)_/¯
        hover_text = (
            f"{canon.obj_id[i]}<br>"
            f"e: {canon.e[i]:.4f}<br>"
            f"i: {np.rad2deg(canon.inc[i]):.2f}°<br>"
            f"Ω: {np.rad2deg(canon.node[i]):.2f}°<br>"
            f"ω: {np.rad2deg(canon.argp[i]):.2f}°"
        )

        # these are being extracted so we can crudely classify them later
        L = float(canon.L[i])
        e = float(canon.e[i])
        inc = float(canon.inc[i])
        if abs(1 - e**2) < 1e-12:  # <-- protect against parabolic orbits
            a = np.inf
        else:
            a = L / (1 - e**2)
        if (not np.isfinite(a)) or (a == 0.0):  # <-- same again
            Tj = np.nan
        else:
            Tj = (5.2044 / a) + 2.0 * np.cos(inc) * np.sqrt((a / 5.2044) * (1 - e**2))
        pop = classify(a, e, Tj)

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(color="rgba(144, 167, 209, 0.7)", width=3),
                hovertemplate=hover_text + "<extra></extra>",
                showlegend=False,
                name=str(canon.obj_id[i]),
                meta={"kind": pop},
            )
        )

    # -- plot input object epoch position --
    if orbit_pos is not None:
        # want to get a hover label with epoch info so first
        # extract from conic objects
        mjd = np.asarray(canon.epochMJD_TDB, dtype=float)
        mjd_str = np.array([f"{m:.5f}" for m in mjd], dtype="U32")

        # also good to have it in YYYY MM DD format
        ymd_str = np.empty(mjd.shape[0], dtype="U10")
        base = datetime(1858, 11, 17)  # <-- MJD 0
        for i, mjd in enumerate(mjd):
            dt = base + timedelta(days=float(mjd))
            ymd_str[i] = dt.strftime("%Y %b %d")

        # vectorise it by stacking all labels together to add
        # as one trace into plotly
        mjd_stack = np.column_stack([mjd_str, ymd_str])
        hover_text = "%{text}<br>" "@ MJD %{customdata[0]}<br>" "(%{customdata[1]})" "<extra></extra>"

        fig.add_trace(
            go.Scatter3d(
                x=orbit_pos[:, 0],
                y=orbit_pos[:, 1],
                z=orbit_pos[:, 2],
                mode="markers",
                marker=dict(size=3.5, color="rgba(255,255,255,0.9)"),
                showlegend=False,
                meta={"kind": "epoch"},
                text=np.asarray(canon.obj_id),
                customdata=mjd_stack,
                hovertemplate=hover_text,
            )
        )

    # -- plot the sun --
    if plot_sun:
        if sun_xyz is None:
            sx, sy, sz = 0.0, 0.0, 0.0
        else:
            sx, sy, sz = float(sun_xyz[0]), float(sun_xyz[1]), float(sun_xyz[2])

        fig.add_trace(
            go.Scatter3d(
                x=[sx],
                y=[sy],
                z=[sz],
                mode="markers",
                marker=dict(size=6, color="yellow"),
                showlegend=False,
                hovertext="Sun",
            )
        )

    # -- pretty up the figure --
    fig.update_layout(
        template=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        autosize=True,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis=dict(
                title="X [AU]",
                showbackground=False,
                gridcolor="rgba(255, 255, 255, 0.02)",
                zerolinecolor="rgba(255, 255, 255, 0.4)",
            ),
            yaxis=dict(
                title="Y [AU]",
                showbackground=False,
                gridcolor="rgba(255, 255, 255, 0.02)",
                zerolinecolor="rgba(255, 255, 255, 0.4)",
            ),
            zaxis=dict(
                title="Z [AU]",
                showbackground=False,
                gridcolor="rgba(255, 255, 255, 0.02)",
                zerolinecolor="rgba(255, 255, 255, 0.4)",
            ),
            aspectmode="data",
            camera=dict(
                center=dict(
                    x=0, y=0, z=-0.25
                )  # <-- for some reason default camera looks really low to me, negative z shifts scene up in the frame
            ),
        ),
    )

    # return figure object if user wants
    if return_fig:
        return fig

    # otherwise write out and show
    if output:
        fig.write_html(output)
        fig.show()

    fig.show()


# --- make the dash app ---
def run_dash_app(fig2d_cache: dict[tuple[str, str], "object"], fig3d_cache: dict[tuple[str, str], "object"]):
    """
    Create and
    """
    import dash
    from dash import Dash, dcc, html, Input, Output, State, ctx
    import dash_daq as daq
    import dash_ag_grid as dag
    import copy
    import os
    import threading
    import webbrowser

    # set up app and (very optionally) link to latex stylesheet
    app = Dash(
        __name__,
        assets_folder="data",
        external_stylesheets=["https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"],
    )

    # establsh day/night mode theming
    THEME = {
        "night": {
            "bg": "black",
            "fg": "white",
            "muted": "rgba(255,255,255,0.82)",
            "grid": "rgba(255,255,255,0.08)",
            "zero": "rgba(255,255,255,0.12)",
            "drawer_bg": "rgba(0,0,0,0.70)",
            "tab_bg": "rgba(0,0,0,0.55)",
            "border": "rgba(255,255,255,0.18)",
        },
        "day": {
            "bg": "white",
            "fg": "#111111",
            "muted": "rgba(0,0,0,0.75)",
            "grid": "rgba(0,0,0,0.10)",
            "zero": "rgba(0,0,0,0.18)",
            "drawer_bg": "rgba(255,255,255,0.86)",
            "tab_bg": "rgba(255,255,255,0.75)",
            "border": "rgba(0,0,0,0.15)",
        },
    }

    ORBIT_COLOUR = {"night": "rgba(144,167,209,0.70)", "day": "rgba(30,60,120,0.85)"}

    # defaults
    night_default = True
    theme0 = THEME["night"] if night_default else THEME["day"]
    fg0 = theme0["fg"]

    origin_bary_default = False
    plane_equ_default = False
    view_3d_default = True
    opacity_default = 0.5

    default_key = ("helio", "ecl")
    initial_fig = fig3d_cache.get(default_key) or next(iter(fig3d_cache.values()))

    # in order to make a selecter for the orbits, we build an inventory of all selectable objects
    def collect_inventory(figs: list[object]):
        planets: set[str] = set()
        objids: set[str] = set()
        kinds: dict[str, str] = {}

        for f in figs:
            for tr in getattr(f, "data", []) or []:
                meta = getattr(tr, "meta", None)
                name = str(getattr(tr, "name", ""))

                if not name or name == "ref-plane":
                    continue

                if isinstance(meta, dict):
                    k = meta.get("kind")  # <-- finally the meta tags come in to play!

                    if k == "Planet":
                        planets.add(name)
                        continue

                    if (
                        getattr(tr, "type", None) in ("scatter", "scatter3d")
                        and getattr(tr, "mode", None) == "lines"
                    ):
                        objids.add(name)
                        kinds.setdefault(name, k)

        return sorted(planets), sorted(objids), kinds

    # get our list of selectable things (if for some reason the cache is partial or could be
    # broken, fall back to whatever the initial_fig is)
    inv_planets, inv_objids, orbit_kinds = collect_inventory(list(fig3d_cache.values()))
    if not inv_planets or not inv_objids:
        inv_planets, inv_objids, orbit_kinds = collect_inventory([initial_fig])

    # these are our table inventory rows
    inventory_rows = [{"kind": "Planet", "name": p} for p in inv_planets] + [
        {"kind": orbit_kinds.get(o, "Other"), "name": o} for o in inv_objids
    ]

    # set up title/label styling helpers for swapping day/night, frame, origin, and view modes
    def title_style(fg: str):
        return {"marginBottom": "10px", "fontSize": "20px", "color": fg, "opacity": 0.95}

    def sublabel_style(fg: str):
        return {"fontSize": "17px", "color": fg, "opacity": 0.90}

    # this creates the proper div environment for the toggle switches so the labels wrap well and look nice
    def labeled_toggle(
        left_label: str, toggle_id: str, right_label: str, value: bool, fg: str, width_pix: int = 220
    ):
        return html.Div(
            [
                html.Div(
                    left_label,
                    style={**sublabel_style(fg), "marginRight": "14px", "whiteSpace": "nowrap"},
                    className="toggle-label",
                ),
                daq.ToggleSwitch(id=toggle_id, value=value, size=70, color="#4cd964"),
                html.Div(
                    right_label,
                    style={**sublabel_style(fg), "marginLeft": "14px", "whiteSpace": "nowrap"},
                    className="toggle-label",
                ),
            ],
            style={"display": "flex", "alignItems": "center", "justifyContent": "center", "width": "100%"},
        )

    # set up a template consistent "block" for all controls to be built into
    def control_block(title: str, body, fg: str, width_px: int = 200):
        return html.Div(
            [
                html.Div(title, style={**title_style(fg), "textAlign": "center", "width": "100%"}),
                html.Div(body, style={"display": "flex", "justifyContent": "center", "width": "100%"}),
            ],
            style={
                "width": "100%",
                "minWidth": "0",
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
            },
        )

    # create a special plane toggle block with the slider
    plane_block_layout = control_block(
        "Plane",
        html.Div(
            [
                html.Div(
                    labeled_toggle(
                        "Ecl", "plane-toggle", "Equ", value=plane_equ_default, fg=fg0, width_pix=220
                    ),
                    style={
                        "display": "flex",
                        "justifyContent": "center",
                        "width": "100%",
                        "maxWidth": "260px",
                        "margin": "0 auto 14px auto",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            "Opacity",
                            style={**sublabel_style(fg0), "flex": "1 1 auto", "textAlign": "center"},
                        ),
                        html.Div(
                            "Colour", style={**sublabel_style(fg0), "flex": "0 0 42px", "textAlign": "center"}
                        ),
                    ],
                    style={
                        "width": "100%",
                        "maxWidth": "260px",
                        "display": "flex",
                        "gap": "10px",
                        "margin": "0 auto 6px auto",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            dcc.Slider(
                                id="plane-opacity",
                                min=0.0,
                                max=1.0,
                                step=0.02,
                                value=opacity_default,
                                marks={0.0: "0", 0.5: "0.5", 1.0: "1.0"},
                                className="opacity-slider",
                            ),
                            style={"flex": "1 1 auto", "minWidth": "0"},
                        ),
                        html.Button(
                            "🖌\ufe0e",  # <-- is emoji pasting good coding practice?
                            id="plane-colour-button",
                            n_clicks=0,
                            style={
                                "flex": "0 0 42px",
                                "width": "42px",
                                "height": "34px",
                                "borderRadius": "10px",
                                "border": f"1px solid {theme0['border']}",
                                "background": theme0["tab_bg"],
                                "color": fg0,
                                "cursor": "pointer",
                                "padding": "0",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                            },
                            title="Edit Colour",
                        ),
                    ],
                    style={
                        "width": "100%",
                        "maxWidth": "260px",
                        "margin": "0 auto",
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "10px",
                    },
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "gap": "10px",
                "width": "100%",
                "minWidth": "0",
            },
        ),
        fg=fg0,
    )

    # create a special switch for 2D plots to choose which panels to display and how many
    panel_mode_block = html.Div(
        [
            html.Div(
                "2D Panel Controls",
                style={**sublabel_style(fg0), "textAlign": "center", "marginBottom": "6px"},
            ),
            dcc.RadioItems(
                id="panel-mode",
                options=[
                    {"label": "1 panel", "value": "single"},
                    {"label": "2 panels", "value": "double"},
                ],
                value="double",
                inline=True,
                style={"display": "flex", "justifyContent": "center", "gap": "18px"},
                labelStyle={**sublabel_style(fg0), "cursor": "pointer"},
            ),
        ],
        style={
            "width": "220px",
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "marginTop": "14px",
        },
    )

    # -- create entire layout of page --
    app.layout = html.Div(
        [
            # these store camera/zoom info across theme/origin/plane changes, whether the
            # controls drawer is open or not, and the height of the drawer
            dcc.Store(id="view-state", data={"mode": None, "camera": None, "xrange": None, "yrange": None}),
            dcc.Store(id="controls-open", data=False),
            dcc.Store(id="drawer-height", data=0),
            # our visible object sets
            dcc.Store(id="visible-objids", data=inv_objids),
            dcc.Store(id="visible-planets", data=inv_planets),
            # here's our button to toggle down the settings menu
            html.Button(
                "⌄",
                id="controls-tab",
                n_clicks=0,
                style={
                    "position": "fixed",
                    "top": "10px",
                    "left": "50%",
                    "transform": "translateX(-50%)",
                    "zIndex": 10000,
                    "pointerEvents": "auto",
                    "width": "64px",
                    "height": "36px",
                    "borderRadius": "14px",
                    "border": f"1px solid {theme0['border']}",
                    "background": theme0["tab_bg"],
                    "color": fg0,
                    "fontSize": "26px",
                    "cursor": "pointer",
                    "backdropFilter": "blur(10px)",
                    "transition": "top 180ms ease",
                },
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("", style={"justifySelf": "start"}),
                                    html.Div(
                                        "Controls",
                                        style={
                                            "fontSize": "22px",
                                            "fontWeight": 600,
                                            "justifySelf": "center",
                                        },
                                    ),
                                    html.Button(
                                        "x",
                                        id="controls-close",
                                        n_clicks=0,
                                        style={
                                            "justifySelf": "end",
                                            "border": "none",
                                            "background": "transparent",
                                            "color": "inherit",
                                            "fontSize": "22px",
                                            "cursor": "pointer",
                                        },
                                    ),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "1fr auto 1fr",
                                    "alignItems": "center",
                                    "marginBottom": "14px",
                                },
                            ),
                            html.Div(
                                html.Div(
                                    [
                                        control_block(
                                            "Theme",
                                            labeled_toggle(
                                                "Day",
                                                "theme-toggle",
                                                "Night",
                                                value=night_default,
                                                fg=fg0,
                                                width_pix=220,
                                            ),
                                            fg=fg0,
                                            width_px=220,
                                        ),
                                        control_block(
                                            "Origin",
                                            labeled_toggle(
                                                "Helio",
                                                "origin-toggle",
                                                "Bary",
                                                value=origin_bary_default,
                                                fg=fg0,
                                                width_pix=220,
                                            ),
                                            fg=fg0,
                                            width_px=220,
                                        ),
                                        plane_block_layout,
                                        html.Div(
                                            [
                                                control_block(
                                                    "View",
                                                    labeled_toggle(
                                                        "2D",
                                                        "view-toggle",
                                                        "3D",
                                                        value=view_3d_default,
                                                        fg=fg0,
                                                        width_pix=200,
                                                    ),
                                                    fg=fg0,
                                                    width_px=220,
                                                ),
                                                html.Div(
                                                    panel_mode_block,
                                                    id="panel-controls-wrapper",
                                                    style={
                                                        "display": "none"
                                                    },  # <-- start hidden since default view is 3D
                                                ),
                                            ],
                                            style={
                                                "display": "flex",
                                                "flexDirection": "column",
                                                "alignItems": "center",
                                                "gap": "10px",
                                                "width": "100%",
                                                "minWidth": "0",
                                            },
                                        ),
                                        control_block(
                                            "Edit Objects",
                                            html.Div(
                                                html.Button(
                                                    [html.Span("✎", style={"marginRight": "10px"}), "Edit"],
                                                    id="objects-open",
                                                    n_clicks=0,
                                                    style={
                                                        "width": "180px",
                                                        "height": "44px",
                                                        "borderRadius": "14px",
                                                        "border": f"1px solid {theme0['border']}",
                                                        "background": theme0["tab_bg"],
                                                        "color": fg0,
                                                        "fontSize": "20px",
                                                        "fontWeight": 600,
                                                        "cursor": "pointer",
                                                    },
                                                ),
                                                style={"marginTop": "6px"},
                                            ),
                                            fg=fg0,
                                            width_px=220,
                                        ),
                                    ],
                                    id="controls-row",
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "repeat(5, minmax(260px, 1fr))",  # <-- this makes it an evenly spaced grid of 5 objects
                                        "alignItems": "start",
                                        "gap": "32px",
                                        "width": "100%",
                                        "boxSizing": "border-box",
                                        "justifyItems": "stretch",
                                    },
                                ),
                                style={
                                    "width": "100%",
                                    "overflowX": "auto",
                                    "overflowY": "visible",
                                    "paddingLeft": "32px",
                                    "paddingRight": "32px",
                                    "boxSizing": "border-box",
                                },
                            ),
                        ],
                        id="controls-content",
                        style={
                            "padding": "18px",
                            "paddingBottom": "64px",
                            "boxSizing": "border-box",
                            "background": theme0["drawer_bg"],
                            "color": fg0,
                            "borderBottom": f"1px solid {theme0['border']}",
                            "backdropFilter": "blur(10px)",
                        },
                    )
                ],
                id="controls-drawer",
                style={
                    "position": "fixed",
                    "top": "0",
                    "left": "0",
                    "right": "0",
                    "zIndex": 4000,
                    "background": "rgba(0,0,0,0)",
                    "borderBottom": "none",
                    "backdropFilter": "none",
                    "overflow": "visible",
                    "display": "flex",
                    "flexDirection": "column",
                    "transform": "translateY(-100%)",
                    "transition": "transform 180ms ease",
                    "pointerEvents": "none",
                },
            ),
            html.Div(
                [
                    # this is the actual plot being drawn
                    dcc.Graph(
                        id="orbit-graph",
                        figure=initial_fig,
                        style={"width": "100%", "height": "100%"},
                        config={"displaylogo": False, "responsive": True, "displayModeBar": True},
                    ),
                    # panel dropdown overlay (on-plot, not in drawer)
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="panel-single",
                                options=[{"label": p, "value": p} for p in ["XY", "XZ", "YZ"]],
                                value="XY",
                                clearable=False,
                                style={"width": "110px", "color": "#111111"},
                            ),
                        ],
                        id="panel-overlay-single",
                        style={
                            "position": "absolute",
                            "left": "50%",
                            "top": "10px",
                            "transform": "translateX(-50%)",
                            "zIndex": 2000,
                            "display": "none",
                            "padding": "10px 12px",
                            "borderRadius": "14px",
                            "background": theme0["tab_bg"],
                            "border": f"1px solid {theme0['border']}",
                            "backdropFilter": "blur(10px)",
                            "pointerEvents": "auto",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="panel-left",
                                options=[{"label": p, "value": p} for p in ["XY", "XZ", "YZ"]],
                                value="XY",
                                clearable=False,
                                style={"width": "110px", "color": "#111111"},
                            ),
                        ],
                        id="panel-overlay-left",
                        style={
                            "position": "absolute",
                            "left": "25%",
                            "top": "10px",
                            "transform": "translateX(-50%)",
                            "zIndex": 2000,
                            "display": "none",
                            "padding": "10px 12px",
                            "borderRadius": "14px",
                            "background": theme0["tab_bg"],
                            "border": f"1px solid {theme0['border']}",
                            "backdropFilter": "blur(10px)",
                            "pointerEvents": "auto",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="panel-right",
                                options=[{"label": p, "value": p} for p in ["XY", "XZ", "YZ"]],
                                value="YZ",
                                clearable=False,
                                style={"width": "110px", "color": "#111111"},
                            )
                        ],
                        id="panel-overlay-right",
                        style={
                            "position": "absolute",
                            "left": "75%",
                            "top": "10px",
                            "transform": "translateX(-50%)",
                            "zIndex": 2000,
                            "display": "none",
                            "padding": "10px 12px",
                            "borderRadius": "14px",
                            "background": theme0["tab_bg"],
                            "border": f"1px solid {theme0['border']}",
                            "backdropFilter": "blur(10px)",
                            "pointerEvents": "auto",
                        },
                    ),
                ],
                style={"position": "fixed", "inset": "0", "zIndex": 1},
            ),
            # make the object table modal manager
            html.Div(
                id="objects-modal",
                style={
                    "display": "none",
                    "position": "fixed",
                    "inset": 0,
                    "zIndex": 20000,
                    "background": "rgba(0,0,0,0.55)",
                    "backdropFilter": "blur(6px)",
                    "pointerEvents": "auto",
                },
                children=[
                    html.Div(
                        id="objects-modal-card",
                        style={
                            "width": "min(1200px, 96vw)",
                            "height": "min(820px, 92vh)",
                            "margin": "4vh auto",
                            "background": theme0["drawer_bg"],
                            "border": f"1px solid {theme0['border']}",
                            "borderRadius": "16px",
                            "padding": "16px",
                            "boxSizing": "border-box",
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "12px",
                        },
                        children=[
                            html.Div(
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "space-between",
                                    "gap": "10px",
                                },
                                children=[
                                    html.Div(
                                        "Object Manager",
                                        style={"fontSize": "22px", "fontWeight": 700, "color": fg0},
                                    ),
                                    html.Button(
                                        "x",
                                        id="objects-close",
                                        n_clicks=0,
                                        style={
                                            "border": "none",
                                            "background": "transparent",
                                            "color": fg0,
                                            "fontSize": "22px",
                                            "cursor": "pointer",
                                        },
                                    ),
                                ],
                            ),
                            html.Div(
                                style={
                                    "display": "flex",
                                    "gap": "10px",
                                    "flexWrap": "wrap",
                                    "alignItems": "center",
                                },
                                children=[
                                    dcc.Input(
                                        id="obj-quickfilter",
                                        type="text",
                                        placeholder="Search... (filter rows)",
                                        value="",
                                        style={
                                            "width": "340px",
                                            "height": "40px",
                                            "borderRadius": "12px",
                                            "border": f"1px solid {theme0['border']}",
                                            "padding": "0 12px",
                                            "outline": "none",
                                        },
                                    ),
                                    html.Div(
                                        id="obj-count",
                                        style={"color": fg0, "opacity": 0.9, "fontSize": "14px"},
                                    ),
                                ],
                            ),
                            html.Div(
                                style={
                                    "display": "flex",
                                    "gap": "10px",
                                    "flexWrap": "wrap",
                                    "alignItems": "center",
                                },
                                children=[
                                    html.Button("Show selected", id="obj-show-selected", n_clicks=0),
                                    html.Button("Hide selected", id="obj-hide-selected", n_clicks=0),
                                    html.Button("Show filtered", id="obj-show-filtered", n_clicks=0),
                                    html.Button("Hide filtered", id="obj-hide-filtered", n_clicks=0),
                                    html.Button("Show all", id="obj-show-all", n_clicks=0),
                                    html.Button("Hide all", id="obj-hide-all", n_clicks=0),
                                    html.Button("Invert Selection", id="obj-invert", n_clicks=0),
                                ],
                            ),
                            html.Div(
                                style={"flex": "1 1 auto", "minHeight": "0"},
                                children=[
                                    dag.AgGrid(
                                        id="objects-grid",
                                        columnDefs=[
                                            {
                                                "headerName": "Kind",
                                                "field": "kind",
                                                "width": 120,
                                                "filter": True,
                                            },
                                            {
                                                "headerName": "Name",
                                                "field": "name",
                                                "flex": 1,
                                                "filter": True,
                                                "checkboxSelection": True,
                                                "headerCheckboxSelection": True,
                                            },
                                            {
                                                "headerName": "Visible",
                                                "field": "visible",
                                                "width": 120,
                                                "filter": True,
                                            },
                                        ],
                                        rowData=[],
                                        defaultColDef={"sortable": True, "resizable": True},
                                        dashGridOptions={
                                            "rowSelection": "multiple",
                                            "animateRows": False,
                                            "suppressRowClickSelection": False,
                                            "rowHeight": 34,
                                        },
                                        className="ag-theme-quartz",
                                        style={"height": "100%", "width": "100%"},
                                    )
                                ],
                            ),
                        ],
                    )
                ],
            ),
            # make the colour selector window
            dcc.Store(id="plane-colour-open", data=False),
            html.Div(
                id="plane-colour-modal",
                n_clicks=0,
                style={
                    "display": "none",
                    "position": "fixed",
                    "inset": 0,
                    "zIndex": 25000,
                    "background": "rgba(0,0,0,0.55)",
                    "backdropFilter": "blur(6px)",
                    "pointerEvents": "auto",
                },
                children=[
                    html.Div(
                        id="plane-colour-card",
                        n_clicks=0,
                        style={
                            "width": "min(520px, 92vw)",
                            "margin": "12vh auto",
                            "background": theme0["drawer_bg"],
                            "border": f"1px solid {theme0['border']}",
                            "borderRadius": "16px",
                            "padding": "16px",
                            "boxSizing": "border-box",
                        },
                        children=[
                            html.Div(
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "space-between",
                                },
                                children=[
                                    html.Div(
                                        "Reference Plane Colour",
                                        style={"fontSize": "18px", "fontWeight": 700, "color": fg0},
                                    ),
                                    html.Button(
                                        "x",
                                        id="plane-colour-close",
                                        n_clicks=0,
                                        style={
                                            "border": "none",
                                            "background": "transparent",
                                            "color": fg0,
                                            "fontSize": "22px",
                                            "cursor": "pointer",
                                        },
                                    ),
                                ],
                            ),
                            html.Div(
                                style={"marginTop": "12px", "display": "flex", "justifyContent": "center"},
                                children=[
                                    daq.ColorPicker(id="plane-colour", value={"hex": "#F5B277"}, size=220)
                                ],
                            ),
                        ],
                    )
                ],
            ),
        ],
        id="app-shell",
        className="theme-dark" if night_default else "theme-light",
        style={
            "height": "100vh",
            "width": "100vw",
            "margin": "0",
            "padding": "0",
            "backgroundColor": theme0["bg"],
            "color": fg0,
        },
    )

    # -- interactive --
    # this bit does a bit of javascript-y hackery: basically 1) whenever the controls-open state
    # changes, we grab its id, then 2) get its pixel dimensions on screen (r = ...), then 3) we
    # get that height (or 0 if missing) and store it in drawer-height - all of this is done in web
    app.clientside_callback(
        """
        function(is_open) {
            const el = document.getElementById("controls-drawer");
            if (!el) return 0;
            const r = el.getBoundingClientRect();
            return (r && r.height) ? r.height : 0;
        }
        """,
        Output("drawer-height", "data"),
        Input("controls-open", "data"),
    )

    # this bit lets the code know what state the dropdown controls tab is in (open/closed?)
    @app.callback(
        Output("controls-open", "data"),
        Input("controls-tab", "n_clicks"),  # <-- toggle tab open/close
        Input("controls-close", "n_clicks"),  # <-- force close
        State("controls-open", "data"),
        prevent_initial_call=True,
    )
    def toggle_controls(_tab_clicks: int | None, _close_clicks: int | None, is_open: bool):
        trig = (
            ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
        )  # <-- this registers what input fires the callback
        if trig == "controls-close":
            return False  # <-- if the X button is pressed, force closed
        return not bool(
            is_open
        )  # <-- if the \/ button is pressed, invert whatever state is_open is in (ie close/open)

    # now we will alter the position, state, movement, and buttons on the controls drawer here
    @app.callback(
        Output("controls-drawer", "style"),
        Output("controls-tab", "children"),
        Output("controls-tab", "style"),
        Input("controls-open", "data"),
        Input("theme-toggle", "value"),
        Input("drawer-height", "data"),
        State("controls-drawer", "style"),
        State("controls-tab", "style"),
    )
    def style_drawer(
        is_open: bool,
        night_mode: bool,
        drawer_h: float | int | None,
        drawer_style: dict[str, Any] | None,
        tab_style: dict[str, Any] | None,
    ):
        theme = THEME["night"] if night_mode else THEME["day"]
        fg = theme["fg"]

        # alter some aspects of the drawer (note: in this context a drawer is the slidey box
        # that comes down when we press the \/ button). note that the style differs if it is
        # open or closed
        ds = dict(drawer_style or {})
        ds["background"] = "rgba(0,0,0,0)"
        ds["borderBottom"] = "none"
        ds["backdropFilter"] = "none"
        ds["overflow"] = "visible"
        ds["display"] = "flex"
        ds["flexDirection"] = "column"
        ds["transition"] = "transform 180ms ease"  # <-- make it smooth pulling out
        ds["transform"] = "translateY(0%)" if is_open else "translateY(-100%)"
        ds["zIndex"] = 4000
        ds["pointerEvents"] = (
            "auto" if is_open else "none"
        )  # <-- makes sure it wont block the plotly plot elements when closed

        # alter some aspects of the button (note: in this context its also called a tab). the
        # colour and theme changes depending on day/night mode
        ts = dict(tab_style or {})
        ts["position"] = "fixed"
        ts["left"] = "50%"
        ts["transform"] = "translateX(-50%)"
        ts["zIndex"] = 10000
        ts["pointerEvents"] = "auto"
        ts["width"] = "64px"
        ts["height"] = "36px"
        ts["cursor"] = "pointer"
        ts["fontSize"] = "26px"
        ts["borderRadius"] = "14px"
        ts["background"] = theme["tab_bg"]
        ts["color"] = fg
        ts["border"] = f"1px solid {theme['border']}"
        ts["backdropFilter"] = "blur(10px)"
        ts["transition"] = "top 180ms ease"

        # this bit places the \/ button either near the top (10px) when closed, or near the bottom edge
        # of the drawer (inset top_px) when open
        try:
            h = float(drawer_h) if drawer_h is not None else 0.0
        except Exception:
            h = 0.0

        if is_open and h > 0:
            top_px = max(
                10, int(h - 36 - 10)
            )  # <-- drawer bottom (h) - button height (36) - small padding (10)
            ts["top"] = f"{top_px}px"
        else:
            ts["top"] = "10px"

        chevron = "⌃" if is_open else "⌄"

        return ds, chevron, ts

    # reference plane colour picker
    @app.callback(
        Output("plane-colour-modal", "style", allow_duplicate=True),
        Output("plane-colour-card", "style"),
        Input("plane-colour-button", "n_clicks"),
        Input("plane-colour-close", "n_clicks"),
        Input("theme-toggle", "value"),
        State("plane-colour-modal", "style"),
        State("plane-colour-card", "style"),
        prevent_initial_call=True,
    )
    def toggle_plane_colour_modal(
        open_clicks: int | None,
        close_clicks: int | None,
        night_mode: bool,
        modal_style: dict[str, Any] | None,
        card_style: dict[str, Any] | None,
    ):
        theme = THEME["night"] if night_mode else THEME["day"]
        ms = dict(modal_style or {})
        cs = dict(card_style or {})

        # keep theme in sync
        cs["background"] = theme["drawer_bg"]
        cs["border"] = f"1px solid {theme['border']}"

        trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
        if trig == "plane-colour-button":
            ms["display"] = "block"
        elif trig == "plane-colour-close":
            ms["display"] = "none"

        return ms, cs

    # add ability to close out of colour picker by just selecting backdrop
    @app.callback(
        Output("plane-colour-modal", "style", allow_duplicate=True),
        Input("plane-colour-modal", "n_clicks"),
        Input("plane-colour-card", "n_clicks"),
        State("plane-colour-modal", "style"),
        prevent_initial_call=True,
    )
    def close_plane_colour_on_backdrop(
        modal_clicks: int | None, card_clicks: int | None, style: dict[str, Any] | None
    ):
        trig = ctx.triggered_id

        # if the backdrop itself was clicked, then close altogether
        if trig == "plane-colour-modal":
            style = dict(style or {})
            style["display"] = "none"
            return style

        # but if click was inside the colour picker card, do nothing
        raise dash.exceptions.PreventUpdate

    # object manager callback stuff (open/close + theming)
    @app.callback(
        Output("objects-modal", "style"),
        Output("objects-modal-card", "style"),
        Input("objects-open", "n_clicks"),
        Input("objects-close", "n_clicks"),
        Input("theme-toggle", "value"),
        State("objects-modal", "style"),
        State("objects-modal-card", "style"),
        prevent_initial_call=True,
    )
    def toggle_objects_modal(
        open: int | None,
        close: int | None,
        night_mode: bool,
        modal_style: dict[str, Any] | None,
        card_style: dict[str, Any] | None,
    ):
        theme = THEME["night"] if night_mode else THEME["day"]
        fg = theme["fg"]

        ms = dict(modal_style or {})
        cs = dict(card_style or {})

        # keep theme in sync even if modal already open
        cs["background"] = theme["drawer_bg"]
        cs["border"] = f"1px solid {theme['border']}"

        trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
        if trig == "objects-open":
            ms["display"] = "block"
        elif trig == "objects-close":
            ms["display"] = "none"

        return ms, cs

    # add search filtering
    @app.callback(Output("objects-grid", "quickFilterText"), Input("obj-quickfilter", "value"))
    def grid_quickfilter(txt: str | None):
        return txt or ""

    # make orbits visible / invisibile from table selections
    @app.callback(
        Output("objects-grid", "rowData"),
        Output("obj-count", "children"),
        Input("visible-objids", "data"),
        Input("visible-planets", "data"),
    )
    def sync_grid_rows(visible_objids: list[str] | None, visible_planets: list[str] | None):
        vis_o = set(visible_objids or [])
        vis_p = set(visible_planets or [])

        rows = []
        for r in inventory_rows:
            k = r["kind"]
            name = r["name"]
            if k == "Planet":
                vis = name in vis_p
            else:
                vis = name in vis_o
            rows.append({"kind": k, "name": name, "visible": "✓" if vis else ""})

        shown = len(vis_o) + len(vis_p)
        total = len(inv_objids) + len(inv_planets)
        return rows, f"Visible: {shown} / {total}"

    # bulk actions from the manager
    @app.callback(
        Output("visible-objids", "data"),
        Output("visible-planets", "data"),
        Input("obj-show-selected", "n_clicks"),
        Input("obj-hide-selected", "n_clicks"),
        Input("obj-show-filtered", "n_clicks"),
        Input("obj-hide-filtered", "n_clicks"),
        Input("obj-show-all", "n_clicks"),
        Input("obj-hide-all", "n_clicks"),
        Input("obj-invert", "n_clicks"),
        State("objects-grid", "selectedRows"),
        State("objects-grid", "virtualRowData"),
        State("visible-objids", "data"),
        State("visible-planets", "data"),
        prevent_initial_call=True,
    )
    def bulk_visibility(
        n1: int | None,
        n2: int | None,
        n3: int | None,
        n4: int | None,
        n5: int | None,
        n6: int | None,
        n7: int | None,
        selected_rows: list[dict[str, Any]] | None,
        virtual_rows: list[dict[str, Any]] | None,
        visible_objids: list[str] | None,
        visible_planets: list[str] | None,
    ):
        # /\ all of the n* are the various click values for the inputs, if they're not there plotly and dash break
        trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

        vis_o = set(visible_objids or [])
        vis_p = set(visible_planets or [])

        def split(rows: list[dict[str, Any]] | None):
            rows = rows or []
            ps = {
                str(r.get("name"))
                for r in rows
                if str(r.get("kind")) == "Planet" and r.get("name") is not None
            }
            os = {
                str(r.get("name"))
                for r in rows
                if str(r.get("kind")) != "Planet" and r.get("name") is not None
            }
            return ps, os

        sel_p, sel_o = split(selected_rows)
        fil_p, fil_o = split(virtual_rows)

        # joe learns logical operators:
        if trig == "obj-show-selected":
            vis_p |= sel_p
            vis_o |= sel_o
        elif trig == "obj-hide-selected":
            vis_p -= sel_p
            vis_o -= sel_o
        elif trig == "obj-show-filtered":
            vis_p |= fil_p
            vis_o |= fil_o
        elif trig == "obj-hide-filtered":
            vis_p -= fil_p
            vis_o -= fil_o
        elif trig == "obj-show-all":
            vis_p = set(inv_planets)
            vis_o = set(inv_objids)
        elif trig == "obj-hide-all":
            vis_p = set()
            vis_o = set()
        elif trig == "obj-invert":
            vis_p = set(inv_planets) - vis_p
            vis_o = set(inv_objids) - vis_o
        else:
            raise dash.exceptions.PreventUpdate

        return sorted(vis_o), sorted(vis_p)

    # this section records the zoom/camera state into view-state
    @app.callback(
        Output("view-state", "data"),
        Input("orbit-graph", "relayoutData"),
        State("view-toggle", "value"),
        State("view-state", "data"),
        prevent_initial_call=True,
    )
    def capture_view(relayout: dict[str, Any] | None, view_3d: bool, state: dict[str, Any] | None):
        # ignore any callbacks that don't actually change the view
        if not relayout:
            return dash.no_update

        # ignore relayouts that are just layout/style updates
        camera_keys = {
            "scene.camera",
            "xaxis.range[0]",
            "xaxis.range[1]",
            "yaxis.range[0]",
            "yaxis.range[1]",
            "xaxis.autorange",
            "yaxis.autorange",
        }

        if not any(k in relayout for k in camera_keys):
            return dash.no_update

        # figure out what mode we're in
        state = dict(state or {})
        mode = "3d" if view_3d else "2d"

        # if the mode has changed since last time, reset the stored view
        # (viewpoint between 2d <-> 3d shouldn't be saved)
        if state.get("mode") != mode:
            state = {"mode": mode, "camera": None, "xrange": None, "yrange": None}
        state["mode"] = mode

        # if we're in 3d and the camera exists, store it
        if mode == "3d":
            cam = relayout.get("scene.camera")
            if cam:
                state["camera"] = cam
        # however if we're in 2d, store the x/y axes ranges
        else:
            x0 = relayout.get("xaxis.range[0]")
            x1 = relayout.get("xaxis.range[1]")
            y0 = relayout.get("yaxis.range[0]")
            y1 = relayout.get("yaxis.range[1]")

            if x0 is not None and x1 is not None:
                state["xrange"] = [x0, x1]
            if y0 is not None and y1 is not None:
                state["yrange"] = [y0, y1]

            # or if the range is reset due to autoranging, clear stored ranges
            if relayout.get("xaxis.autorange") or relayout.get("yaxis.autorange"):
                state["xrange"] = None
                state["yrange"] = None

        return state

    # here we're gonna update everything that is actual content such as figures, rather
    # than just updating the drawer/toggle mechanics
    @app.callback(
        Output("orbit-graph", "figure"),
        Output("app-shell", "style"),
        Output("app-shell", "className"),
        Output("panel-controls-wrapper", "style"),
        Output("panel-overlay-single", "style"),
        Output("panel-overlay-left", "style"),
        Output("panel-overlay-right", "style"),
        Input("theme-toggle", "value"),
        Input("view-toggle", "value"),
        Input("origin-toggle", "value"),
        Input("plane-toggle", "value"),
        Input("plane-opacity", "value"),
        Input("plane-colour", "value"),
        Input("panel-mode", "value"),
        Input("panel-single", "value"),
        Input("panel-left", "value"),
        Input("panel-right", "value"),
        Input("visible-objids", "data"),
        Input("visible-planets", "data"),
        State("view-state", "data"),
    )
    def update_orbit_plot(
        night_mode: bool,
        view_3d: bool,
        origin_bary: bool,
        plane_equ: bool,
        opacity: float,
        plane_colour,
        panel_mode: str,
        panel_single: str,
        panel_left: str,
        panel_right: str,
        visible_objids: list[str] | None,
        visible_planets: list[str] | None,
        view_state: dict[str, object],
    ):
        # set up some figure themings
        theme = THEME["night"] if night_mode else THEME["day"]
        bg = theme["bg"]
        fg = theme["fg"]
        grid = theme["grid"]
        zero = theme["zero"]
        orbit_colour = ORBIT_COLOUR["night"] if night_mode else ORBIT_COLOUR["day"]

        # set a default plot to show first
        origin = "bary" if origin_bary else "helio"
        plane = "equ" if plane_equ else "ecl"

        # deepcopy important as fig.data is mutable, so altering colours messes up cached figures and so toggles
        if view_3d:
            fig = copy.deepcopy(fig3d_cache[(origin, plane)])
        else:
            if panel_mode == "single":
                fig = copy.deepcopy(fig2d_cache[(origin, plane, "single", panel_single, None)])
            else:
                fig = copy.deepcopy(fig2d_cache[(origin, plane, "double", panel_left, panel_right)])

        for trace in fig.data:
            # we need to differentiate planets from input objects to make sure
            # they are coloured properly, so we use their meta tags we set up
            # earlier to toggle their colours appropriately
            is_planet = (
                getattr(trace, "meta", None) is not None
                and isinstance(trace.meta, dict)
                and trace.meta.get("kind") == "Planet"
            )
            if is_planet:
                name = str(trace.name)
                trace.line.color = (
                    PLANET_COLOURS_NIGHT.get(name, "rgba(220,220,220,0.7)")
                    if night_mode
                    else PLANET_COLOURS_DAY.get(name, "rgba(220,220,220,0.7)")
                )
                continue

            # every other non-planet is the default colour then
            if trace.type in ("scatter", "scatter3d") and getattr(trace, "mode", None) == "lines":
                trace.line.color = orbit_colour

            # also colour input object epoch markers differently if day or night mode
            meta = getattr(trace, "meta", None)
            if isinstance(meta, dict) and meta.get("kind") == "epoch":
                trace.marker.color = "rgba(255,255,255,0.9)" if night_mode else "rgba(30,30,30,0.9)"

        # apply object visibility
        vis_o = set(visible_objids or [])
        vis_p = set(visible_planets or [])

        for trace in fig.data:
            meta = getattr(trace, "meta", None)
            kind = meta.get("kind") if isinstance(meta, dict) else None

            if kind == "Planet":
                trace.visible = str(getattr(trace, "name", "")) in vis_p
                continue

            if (
                getattr(trace, "type", None) in ("scatter", "scatter3d")
                and getattr(trace, "mode", None) == "lines"
            ):
                name = str(getattr(trace, "name", ""))
                if name and name != "ref-plane":
                    trace.visible = name in vis_o

            if isinstance(meta, dict) and meta.get("kind") == "epoch":
                texts = getattr(trace, "text", None)
                texts_list = [] if texts is None else list(texts)

                base = "rgba(255,255,255,0.9)" if night_mode else "rgba(100,100,100,0.9)"
                hidden = "rgba(0,0,0,0)"

                trace.marker.color = [base if str(t) in vis_o else hidden for t in texts_list]

        # if the view is 3d then set the reference plane opacity+colour
        if view_3d:
            for trace in fig.data:
                if getattr(trace, "name", None) == "ref-plane":
                    trace.opacity = opacity

                    if plane_colour and "hex" in plane_colour:
                        c = plane_colour["hex"]
                        trace.update(colorscale=[[0, c], [1, c]], cmin=0, cmax=1, showscale=False)

        # set up the background of the whole figure canvas + plotting region
        # so that they are transparent (ie page theme is controlling background,
        # not plotly itself)
        paper = "rgba(0,0,0,0)" if bg == "black" else "rgba(255,255,255,0)"
        fig.update_layout(paper_bgcolor=paper, plot_bgcolor=paper, font=dict(color=fg), uirevision="keep")

        # style the axis grids in 2d or 3d
        if hasattr(fig, "update_xaxes"):
            fig.update_xaxes(gridcolor=grid, zerolinecolor=zero)
            fig.update_yaxes(gridcolor=grid, zerolinecolor=zero)

        if getattr(fig.layout, "scene", None) is not None:
            fig.update_scenes(
                xaxis_gridcolor=grid,
                xaxis_zerolinecolor=zero,
                yaxis_gridcolor=grid,
                yaxis_zerolinecolor=zero,
                zaxis_gridcolor=grid,
                zaxis_zerolinecolor=zero,
            )

        # get current view state
        view_state = view_state or {}
        mode = "3d" if view_3d else "2d"

        # if we're in the same view mode, reapply the stored view. this helps us
        # preserve the view across theme/origin/plane changes
        if view_state.get("mode") == mode:
            if mode == "3d" and view_state.get("camera") and getattr(fig.layout, "scene", None) is not None:
                fig.update_layout(scene_camera=view_state["camera"])
            if mode == "2d":
                xr = view_state.get("xrange")
                yr = view_state.get("yrange")
                if xr is not None:
                    fig.update_xaxes(range=xr, autorange=False)
                if yr is not None:
                    fig.update_yaxes(range=yr, autorange=False)

        # set the style of the outermost html container
        shell_style = {
            "height": "100vh",
            "width": "100vw",
            "margin": "0",
            "padding": "0",
            "backgroundColor": bg,
            "color": fg,
        }
        shell_class = "theme-dark" if night_mode else "theme-light"

        # show/hide the 2D panel-mode selector under the view toggle
        panel_controls_style = {"display": "none"} if view_3d else {}

        # per-2D panel overlay base style
        base_overlay = {
            "position": "absolute",
            "top": "10px",
            "zIndex": 2000,
            "padding": "10px 12px",
            "borderRadius": "14px",
            "background": theme["tab_bg"],
            "border": f"1px solid {theme['border']}",
            "backdropFilter": "blur(10px)",
            "pointerEvents": "auto",
            "display": "none",
        }

        single_style = dict(base_overlay)
        left_style = dict(base_overlay)
        right_style = dict(base_overlay)

        # positions of dropdown menus
        single_style["left"] = "25%"
        single_style["transform"] = "translateX(-50%)"

        left_style["left"] = "25%"
        left_style["transform"] = "translateX(-50%)"

        right_style["left"] = "75%"
        right_style["transform"] = "translateX(-50%)"

        if not view_3d:
            if panel_mode == "single":
                single_style["display"] = "block"
            else:
                left_style["display"] = "block"
                right_style["display"] = "block"

        return fig, shell_style, shell_class, panel_controls_style, single_style, left_style, right_style

    # auto-open the browser
    def open_browser():
        webbrowser.open_new("http://127.0.0.1:8050/")

    # this is a reloader that prevents us accidentally opening two tabs
    # at the same time (this kept happening to me in debugging),
    # giving the server time to start before the browser visits the url
    # (see https://stackoverflow.com/questions/9449101/how-to-stop-flask-from-initialising-twice-in-debug-mode?)
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1.0, open_browser).start()

    # avoid spam in the terminal
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    # now run the server!
    app.run(debug=False, use_reloader=True)
