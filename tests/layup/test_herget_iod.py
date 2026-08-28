import os
import numpy as np
from numpy.lib import recfunctions as rfn
from numpy.testing import assert_allclose, assert_equal
import layup.utilities.herget_iod as herget
import spiceypy as spice

from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader
from layup.routines import Observation, get_ephem
from layup.utilities.data_processing_utilities import LayupObservatory
from layup.utilities.datetime_conversions import convert_tdb_date_to_julian_date
from layup.orbitfit import _build_sequence
import assist
import rebound
from layup.orbit_maths import build_ephem_and_mus

SPEED_OF_LIGHT_AU_DAY = 173.145


def test_find_mag_to_adjust():
    """Testing that the formula for finding the closest point on a line works by using a trivial case"""

    point = np.array([1, 2])
    line_point1 = np.array([0, 0])
    line_point2 = np.array([1, 0])

    answer = 1
    mag = herget.find_mag_to_adjust(point, line_point1, line_point2)
    assert mag == answer

    # Trying again for the edge case that the point lies on the line
    point = np.array([0, 0])
    answer = 0
    mag = herget.find_mag_to_adjust(point, line_point1, line_point2)
    assert mag == answer


def test_find_new_vel_with_universal_kepler():
    """Start with an object moving away from target and check it changes direction of velocity
    Also check end position ends up closer to the target"""

    pos = np.array([50, 0, 0])
    vel = np.array([-1e-5, -1e-5, -1e-5])
    target = np.array(
        [55, 5, 5, 0, 0, 0]
    )  # Function should attempt to add a positive velocity in any cartesian direction to get to this position

    t1 = 0
    tn = 300

    [vx1, vy1, vz1], [*pos_new, vxn, vyn, vzn] = herget.find_new_vel_with_universal_kepler(
        t1, tn, [*pos, *vel], target
    )

    assert ([vx1, vy1, vz1] > vel).all()
    assert (
        np.sqrt(sum((target[:3] - pos_new) ** 2)) < np.sqrt(sum((target[:3] - pos) ** 2))
    ).all()  # Check the new end-position is closer to the target

    # Run again to check it continues to converge
    [vx1, vy1, vz1], [*pos_new_rerun, vxn, vyn, vzn] = herget.find_new_vel_with_universal_kepler(
        t1, tn, [*pos, vx1, vy1, vz1], target
    )

    assert (np.sqrt(sum((target[:3] - pos_new_rerun) ** 2)) < np.sqrt(sum((target[:3] - pos_new) ** 2))).all()


def test_find_drho(tmpdir):
    """Take an object with a known rho value (from JPL), see if this function approaches the right direction"""

    os.chdir(tmpdir)
    data = CSVDataReader(
        get_test_filepath("2000OK67_ephem.csv"), "csv", primary_id_column_name="provID"
    ).read_rows()

    layup_observatory = LayupObservatory(cache_dir=None)

    # The units of et are seconds (from J2000). This new column is used by
    # data_processing_utilities.obscodes_to_barycentric.
    et_col = np.array([spice.str2et(row["obsTime"]) for row in data], dtype="<f8")
    data = rfn.append_fields(data, "et", et_col, usemask=False, asrecarray=True)

    pos_vel = layup_observatory.obscodes_to_barycentric(data)
    data = rfn.merge_arrays([data, pos_vel], flatten=True, asrecarray=True, usemask=False)

    observations = []
    for d in data:
        o = Observation.from_astrometry_with_id(
            str(d["provID"]),
            d["ra"] * np.pi / 180.0,
            d["dec"] * np.pi / 180.0,
            convert_tdb_date_to_julian_date(d["obsTime"], None),  # Convert obstime to JD TDB
            [d["x"], d["y"], d["z"]],  # Barycentric position
            [d["vx"], d["vy"], d["vz"]],  # Barycentric velocity
        )
        observations.append(o)

    jds = convert_tdb_date_to_julian_date(data["obsTime"])
    sequence = _build_sequence(jds, sep_dt=90.0)

    obs_1 = observations[0]
    r_e_1 = obs_1.observer_position
    rho_hat_1 = np.array(obs_1.rho_hat)
    rho_1 = 40  # this is the magnitude of rho, direction given by rho_hat, initial guess is 40au
    t_1 = obs_1.epoch
    state_1 = [*r_e_1 + rho_1 * rho_hat_1, 0, 0, 0]

    obs_n = observations[-1]
    r_e_n = obs_n.observer_position
    rho_hat_n = np.array(obs_n.rho_hat)
    rho_n = 40  # this is the magnitude of rho, direction given by rho_hat, initial guess is 40au
    t_n = obs_n.epoch
    r_n = r_e_n + rho_n * rho_hat_n

    # Get original epochs so we can light-time correct them each iteration
    epochs = np.zeros(len(observations))
    for i, observation in enumerate(observations):
        epochs[i] = observation.epoch

    for i, observation in enumerate(observations):
        observation.epoch = epochs[i] - ((rho_1) + (rho_n)) / (2 * SPEED_OF_LIGHT_AU_DAY)

    state_1[3:], _ = herget.find_velocity(t_1, t_n, state_1[:3], r_n, 0.001)

    ephem, _, _ = build_ephem_and_mus()
    print(dir(ephem))
    sim = rebound.Simulation()
    ex = assist.Extras(sim, ephem)
    sim.t = t_1 - ephem.jd_ref

    sim.add(x=state_1[0], y=state_1[1], z=state_1[2], vx=state_1[3], vy=state_1[4], vz=state_1[5])

    residuals = []

    for i, observation in enumerate(observations):

        # For this observation, get A and D
        A = observation.a_vec
        D = observation.d_vec

        t = observation.epoch
        sim.integrate(t - ephem.jd_ref)

        r_e = np.array(observation.observer_position)
        r = sim.particles[0].xyz
        rho = r - r_e
        residuals.append(np.dot(rho / np.linalg.norm(rho), A))
        residuals.append(np.dot(rho / np.linalg.norm(rho), D))
    sum_residuals_1 = sum(np.array(residuals) ** 2)

    # call find_drho, check if it reduces the sum of the residuals

    delta_rho1, delta_rhon, state_1 = herget.find_drho(
        observations, t_1, t_n, state_1[:3], r_n, 0.001, ephem, rho_hat_1, rho_hat_n
    )

    # Update rho values
    rho_1 -= delta_rho1
    r_1 = r_e_1 + rho_1 * np.array(rho_hat_1)
    rho_n -= delta_rhon
    r_n = r_e_n + rho_n * np.array(rho_hat_n)

    for i, observation in enumerate(observations):
        observation.epoch = epochs[i] - ((rho_1) + (rho_n)) / (2 * SPEED_OF_LIGHT_AU_DAY)

    state_1[3:], _ = herget.find_velocity(t_1, t_n, state_1[:3], r_n, 0.001)

    sim = rebound.Simulation()
    ex = assist.Extras(sim, ephem)
    sim.t = t_1 - ephem.jd_ref

    sim.add(x=state_1[0], y=state_1[1], z=state_1[2], vx=state_1[3], vy=state_1[4], vz=state_1[5])

    residuals = []

    for i, observation in enumerate(observations):

        # For this observation, get A and D
        A = observation.a_vec
        D = observation.d_vec

        t = observation.epoch
        sim.integrate(t - ephem.jd_ref)

        r_e = np.array(observation.observer_position)
        r = sim.particles[0].xyz
        rho = r - r_e
        residuals.append(np.dot(rho / np.linalg.norm(rho), A))
        residuals.append(np.dot(rho / np.linalg.norm(rho), D))
    sum_residuals_2 = sum(np.array(residuals) ** 2)

    assert sum_residuals_2 < sum_residuals_1


def test_whole_herget_method(tmpdir):
    import subprocess
    from pathlib import Path

    os.chdir(tmpdir)

    test_filename = "2000OK67_ephem.csv"
    input_file = Path(get_test_filepath(test_filename))
    temp_out_file = f"test_output_{input_file.stem}"

    result = subprocess.run(
        ["layup", "orbitfit", str(input_file), "ADES_csv", "-f", "-o", str(temp_out_file), "-i", "herget"]
    )

    assert result.returncode == 0

    result_file = Path(f"{tmpdir}/{temp_out_file}.csv")
    assert result_file.exists

    output_csv_reader = CSVDataReader(str(result_file), "csv", primary_id_column_name="provID")
    output_data = output_csv_reader.read_rows()

    # "True" orbital parameters taken from JPL to compare
    state_true = np.array(
        [
            39.55820757262561,
            5.655976721534595,
            2.754981653420371,
            -4.215749136290368e-04,
            2.526859896941292e-03,
            1.364363007123930e-03,
        ]
    )

    assert_allclose(
        [*state_true],
        [
            output_data["x"][0],
            output_data["y"][0],
            output_data["z"][0],
            output_data["xdot"][0],
            output_data["ydot"][0],
            output_data["zdot"][0],
        ],
        atol=1e-2,
    )
