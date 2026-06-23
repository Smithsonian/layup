import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal

from layup.predict import (
    predict,
    predict_cli,
    _convert_to_sg,
    _get_on_sky_data,
    layup_get_residual_vectors,
    layup_calculate_rates_and_geometry,
)
from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader


@pytest.mark.parametrize(
    "chunk_size, time_step, input_format",
    [
        (5, 0.5, "BCART_EQ"),
        (5, 1, "BCART_EQ"),
        (5, 2, "BCART_EQ"),
        (3, 0.5, "BCART_EQ"),
        (3, 1, "BCART_EQ"),
        (3, 2, "BCART_EQ"),
        (5, 0.5, "COM"),
        (5, 1, "COM"),
        (5, 2, "COM"),
        (3, 0.5, "COM"),
        (3, 1, "COM"),
        (3, 2, "COM"),
    ],
)
def test_predict_cli(tmpdir, chunk_size, time_step, input_format):
    """Test that the predict cli works for a small CSV file."""
    os.chdir(tmpdir)

    start = 2461091.50080075
    end = 2461101.50080075

    class FakeCliArgs:
        def __init__(self, g=None):
            self.primary_id_column_name = "provID"
            self.n = 1
            self.chunk = chunk_size
            self.station = "X05"
            self.sexagesimal = False
            self.onsky_data = False

    # The naming scheme for the test files indicates its orbit format
    test_filename = f"predict_chunk_{input_format}.csv"
    temp_out_file = f"test_output_{os.path.basename(test_filename)}"
    predict_cli(
        cli_args=FakeCliArgs(),
        input_file=get_test_filepath(test_filename),
        start_date=start,
        end_date=end,
        timestep_day=time_step,
        output_file=temp_out_file,
        cache_dir=None,
        configs=None,
    )

    # Verify predict produced an output file
    assert os.path.exists(temp_out_file)
    # Create a new CSV reader to read in our output file
    output_csv_reader = CSVDataReader(temp_out_file, "csv", primary_id_column_name="provID")
    output_data = output_csv_reader.read_rows()

    # Read the input data and get the provID column
    input_csv_reader = CSVDataReader(get_test_filepath(test_filename), "csv", primary_id_column_name="provID")
    input_data = input_csv_reader.read_rows()

    n_uniq_ids = sum([1 if id else 0 for id in set(input_data["provID"])])
    number_of_predictions_per = len(np.arange(start, end + time_step, time_step))

    # ensure that have a prediction for each object at every time step
    assert_equal(len(output_data), n_uniq_ids * number_of_predictions_per)

    assert np.all(output_data["ra_deg"] <= 360.0) and np.all(output_data["ra_deg"] >= 0.0)
    assert np.all(output_data["dec_deg"] <= 90.0) and np.all(output_data["dec_deg"] >= -90.0)

    # Ensure that the epoch_utc column is present and in the correct format
    assert all(isinstance(epoch, str) for epoch in output_data["epoch_UTC"])
    # Validate the first epoch_UTC value has the expectd time
    assert output_data["epoch_UTC"][0] == "2026 FEB 20 00:00:00"
    assert all(len(epoch) == 20 for epoch in output_data["epoch_UTC"])
    # All of our start and end dates for our predictions are in the year 2026
    assert all(epoch.startswith("2026") for epoch in output_data["epoch_UTC"])

    assert all(isinstance(epoch, float) for epoch in output_data["epoch_JD_TDB"])


def test_external_predict(tmpdir):
    """Ensure that we can run predict with data that doesn't have our csq and ndof columns."""

    # this file contains some rows with csq and ndof columns and some without
    # so this should test that all functionality remains the same.
    class FakeCliArgs:
        def __init__(self, g=None):
            self.onsky_data = False

    data = CSVDataReader(
        get_test_filepath("fit_result_file_example.csv"), "csv", primary_id_column_name="provID"
    ).read_rows()

    times = np.arange(2461091.50080075, 2461101.50080075 + 0.5, step=0.5)
    predictions = predict(
        data,
        obscode="X05",
        times=times,
        num_workers=1,
        cache_dir=None,
        primary_id_column_name="provID",
        args=FakeCliArgs(),
    )

    # make sure we generated a prediction for each object at every time step
    n_uniq_ids = sum([1 if id else 0 for id in set(data["provID"])])
    assert len(predictions) == n_uniq_ids * len(times)


def test_predict_output(tmpdir):
    """Compare the output of predict (as run from the command line) against an
    expected output."""

    os.chdir(tmpdir)

    start = "2025 MAY 18 00:00:00"
    test_filename = "holman.csv"
    input_file = Path(get_test_filepath(test_filename))
    temp_out_file = f"test_output_{input_file.stem}"

    result = subprocess.run(
        ["layup", "predict", str(input_file), "-f", "-o", str(temp_out_file), "-s", start]
    )

    assert result.returncode == 0

    result_file = Path(f"{tmpdir}/{temp_out_file}.csv")
    assert result_file.exists

    # Create a new CSV reader to read in our output file
    output_csv_reader = CSVDataReader(str(result_file), "csv", primary_id_column_name="provID")
    output_data = output_csv_reader.read_rows()

    # Read in the known output
    known_output_file = get_test_filepath("holman_expected_predict.csv")
    known_output_csv_reader = CSVDataReader(known_output_file, "csv", primary_id_column_name="provID")
    known_data = known_output_csv_reader.read_rows()

    assert np.all(output_data["epoch_UTC"] == known_data["epoch_UTC"])
    assert np.allclose(output_data["epoch_JD_TDB"], known_data["epoch_JD_TDB"])
    assert np.allclose(output_data["ra_deg"], known_data["ra_deg"])
    assert np.allclose(output_data["dec_deg"], known_data["dec_deg"])
    assert np.allclose(output_data["rho_x"], known_data["rho_x"])
    assert np.allclose(output_data["rho_y"], known_data["rho_y"])
    assert np.allclose(output_data["rho_z"], known_data["rho_z"])

    # holman.csv carries a non-zero orbit covariance, so predict must report a
    # valid 2x2 sky-plane covariance. We check physical properties rather than
    # golden values: the covariance varies at the ~1e-7 level across platforms,
    # and the propagated covariance has no independent reference in the repo.
    # (The full numerical validation lives in
    # test_predict_observational_covariance_matches_refit_scatter.)
    xx = output_data["obs_cov_xx"]
    yy = output_data["obs_cov_yy"]
    xy = output_data["obs_cov_xy"]
    # All entries finite.
    assert np.all(np.isfinite(xx)) and np.all(np.isfinite(yy)) and np.all(np.isfinite(xy))
    # Positive variances and a positive-definite 2x2 (positive determinant).
    assert np.all(xx > 0.0)
    assert np.all(yy > 0.0)
    assert np.all(xx * yy - xy * xy > 0.0)
    # Regression guard for the obs_cov_yy indexing bug, which set the Dec
    # variance to the RA/Dec off-diagonal term (obs_cov_yy == obs_cov_xy).
    assert np.all(yy != xy)

    # The error-ellipse columns must be the eigen-decomposition of that 2x2:
    # semi-major >= semi-minor > 0 and semi-major^2 = larger eigenvalue. This
    # guards the skyplane_cov_to_radec_cov call (previously passed obs_cov_yy
    # and obs_cov_xy in the wrong order and scaled by cos(dec) in degrees).
    a_arcsec = output_data["a_arcsec"]
    b_arcsec = output_data["b_arcsec"]
    assert np.all(np.isfinite(a_arcsec)) and np.all(np.isfinite(b_arcsec))
    assert np.all(a_arcsec >= b_arcsec)
    assert np.all(b_arcsec > 0.0)
    rad_to_arcsec = (180.0 / np.pi) * 3600.0
    lambda_major = 0.5 * (xx + yy + np.sqrt((xx - yy) ** 2 + (2.0 * xy) ** 2))
    assert np.allclose(a_arcsec, np.sqrt(lambda_major) * rad_to_arcsec)

    # Testing the output of the sexagesimal conversion separately

    result = subprocess.run(
        [
            "layup",
            "predict",
            str(input_file),
            "-f",
            "-o",
            str(temp_out_file),
            "-s",
            start,
            "-sg",
        ]
    )

    assert result.returncode == 0

    result_file = Path(f"{tmpdir}/{temp_out_file}.csv")
    assert result_file.exists

    # Create a new CSV reader to read in our output file
    output_csv_reader = CSVDataReader(str(result_file), "csv", primary_id_column_name="provID")
    output_data = output_csv_reader.read_rows()

    # Read in the known output
    known_output_file = get_test_filepath("holman_expected_predict_sg.csv")
    known_output_csv_reader = CSVDataReader(known_output_file, "csv", primary_id_column_name="provID")
    known_data = known_output_csv_reader.read_rows()

    assert (output_data["ra_str_hms"] == known_data["ra_str_hms"]).all() == True
    assert (output_data["dec_str_dms"] == known_data["dec_str_dms"]).all() == True

    # Check the columns have been swapped too
    assert (known_data.dtype.names == output_data.dtype.names) == True


def test_convert_to_sg():
    """Compare the output given by _convert_to_sg() with an expected output, seeing how it handles edge cases."""

    data = CSVDataReader(
        get_test_filepath("known_sexagesimal.csv"), "csv", primary_id_column_name="provID"
    ).read_rows()

    data = _convert_to_sg(data)

    assert (data["ra_str_hms"] == data["ra_str_hms_CHECK"]).all() == True
    assert (data["dec_str_dms"] == data["dec_str_dms_CHECK"]).all() == True


def test_layup_get_residual_vectors():
    """This is a version of the sorcha function _get_residual_vectors with the ability to
    vectorise if an array is passed. Test will compare its output to the sorcha function."""
    from sorcha.ephemeris.simulation_driver import get_residual_vectors

    vectors = np.array([[1, 2, 3], [4, 5, 6], [0.7, 0.8, 0.9]])
    output_A, output_D = layup_get_residual_vectors(vectors)

    for i in range(len(vectors[0])):
        sorcha_A, sorcha_D = get_residual_vectors(vectors.T[i])
        np.allclose(output_A[i], sorcha_A)
        np.allclose(output_D[i], sorcha_D)


def test_layup_calculate_rates_and_geometry():
    """This is a version of the sorcha function calculate_rates_and_geometry with the ability to
    vectorise if an array is passed. Test will compare its output to the sorcha function."""

    from sorcha.ephemeris.simulation_driver import EphemerisGeometryParameters, calculate_rates_and_geometry
    import pandas as pd

    # define pointings
    pointings_filename = "test_pointings.csv"
    pointings_file = Path(get_test_filepath(pointings_filename))
    pointings = pd.read_csv(pointings_file, header=0)
    print(pointings)
    ephem_geom_params = EphemerisGeometryParameters(
        obj_id=np.array(["Holman", "Holman", "Holman", "Holman", "Holman", "Holman"], dtype=object),
        mjd_tai=None,
        rho=np.array(
            [
                [3.74383268, 3.7434049, 3.74297428, 3.74254091, 3.74210503, 3.74166706],
                [-0.49091188, -0.49024155, -0.48957158, -0.48890264, -0.48823533, -0.48757017],
                [-0.30470345, -0.30443051, -0.30415773, -0.3038851, -0.30361262, -0.30334029],
            ]
        ),
        rho_hat=np.array(
            [
                [0.98829964, 0.98832539, 0.98835109, 0.98837671, 0.98840224, 0.98842766],
                [-0.12959127, -0.12943248, -0.12927383, -0.12911548, -0.1289576, -0.1288003],
                [-0.08043584, -0.08037506, -0.08031437, -0.08025375, -0.0801932, -0.08013271],
            ]
        ),
        rho_mag=np.array([3.78815546, 3.78762392, 3.78708975, 3.7865531, 3.78601432, 3.78547386]),
        r_ast=np.array(
            [
                [3.18771634, 3.18787683, 3.18803727, 3.18819768, 3.18835805, 3.18851838],
                [-1.27380146, -1.27350169, -1.27320191, -1.27290212, -1.27260231, -1.27230248],
                [-0.64391493, -0.6437985, -0.64368206, -0.64356562, -0.64344916, -0.6433327],
            ]
        ),
        v_ast=np.array(
            [
                [0.00385183, 0.00385091, 0.00384998, 0.00384906, 0.00384814, 0.00384722],
                [0.00719365, 0.00719402, 0.00719439, 0.00719475, 0.00719512, 0.00719549],
                [0.00279403, 0.00279421, 0.0027944, 0.00279458, 0.00279477, 0.00279495],
            ]
        ),
    )
    output = layup_calculate_rates_and_geometry(pointings, ephem_geom_params)
    for i in range(len(pointings)):
        pointing = pointings.iloc[i]
        geom_param = EphemerisGeometryParameters()
        geom_param.obj_id = ephem_geom_params.obj_id[i]
        geom_param.mjd_tai = None
        geom_param.rho = ephem_geom_params.rho.T[i]
        geom_param.rho_hat = ephem_geom_params.rho_hat.T[i]
        geom_param.rho_mag = ephem_geom_params.rho_mag.T[i]
        geom_param.r_ast = ephem_geom_params.r_ast.T[i]
        geom_param.v_ast = ephem_geom_params.v_ast.T[i]
        print()
        sorcha_output = calculate_rates_and_geometry(pointing, geom_param)
        for j in range(3, len(sorcha_output)):
            print(output[j][i], sorcha_output[j])
            np.allclose(output[j][i], sorcha_output[j])


def test_get_onsky_data_output(tmpdir):
    # Test against the same output from Sorcha
    os.chdir(tmpdir)

    start = "2025 MAY 18 00:00:00"
    test_filename = "holman.csv"
    input_file = Path(get_test_filepath(test_filename))
    temp_out_file = f"test_output_{input_file.stem}"

    result = subprocess.run(
        [
            "layup",
            "predict",
            str(input_file),
            "-f",
            "-o",
            str(temp_out_file),
            "-s",
            start,
            "-osd",
        ]
    )
    assert result.returncode == 0
    result_file = Path(f"{tmpdir}/{temp_out_file}.csv")
    results = pd.read_csv(result_file, header=0)
    # read in the expected output, check if the numbers are close
    expected_filename = "onsky_expected.csv"
    expected_file = Path(get_test_filepath(expected_filename))
    expected = pd.read_csv(expected_file, header=0)
    for column in [
        "Range_LTC_au",
        "Range_LTC_km",
        "RangeRate_LTC_km_s",
        "Obj_Sun_LTC_au",
        "Obj_Sun_LTC_km",
        "Obj_Sun_x_LTC_km",
        "Obj_Sun_y_LTC_km",
        "Obj_Sun_z_LTC_km",
        "Obj_Sun_vx_LTC_km_s",
        "Obj_Sun_vy_LTC_km_s",
        "Obj_Sun_vz_LTC_km_s",
        "phase_deg",
    ]:
        np.allclose(results[column], expected[column])


def _prep_for_fit(obs, observatory):
    """Append the ``et`` column and observer barycentric state that the
    per-object ``_orbitfit`` worker expects (normally added by ``orbitfit``)."""
    import numpy.lib.recfunctions as rfn
    import spiceypy as spice

    et = np.array([spice.str2et(row["obsTime"]) for row in obs], dtype="<f8")
    obs = rfn.append_fields(obs, "et", et, usemask=False, asrecarray=True)
    pos_vel = observatory.obscodes_to_barycentric(obs)
    return rfn.merge_arrays([obs, pos_vel], flatten=True, asrecarray=True, usemask=False)


# (label, observation file, prediction obscode). The interstellar case
# (3I/ATLAS, designation A11pl3Z) is a strongly hyperbolic orbit (e ~ 6.5), so
# it exercises the covariance projection well outside the main-belt regime.
_SCATTER_CASES = [
    ("mainbelt", "1_random_mpc_ADES_provIDs_no_sats_micro.csv", "704"),
    ("interstellar_3I_ATLAS", "3I_ATLAS_ades.csv", "I40"),
]


@pytest.mark.parametrize("label, obs_filename, obscode", _SCATTER_CASES, ids=[c[0] for c in _SCATTER_CASES])
def test_predict_observational_covariance_matches_refit_scatter(label, obs_filename, obscode):
    """Issue #278: Monte-Carlo validation of predict's observational covariance.

    Fit an orbit, predict to an epoch to obtain the on-sky (tangent-plane)
    covariance, then repeatedly perturb the observations by their assumed 1"
    uncertainty, refit, and re-predict. If the orbit covariance and its
    projection to RA/Dec are correct, the scatter of the predicted positions
    must be consistent with the predicted observational covariance: the squared
    Mahalanobis distances follow a chi-square distribution with 2 degrees of
    freedom, so their mean is ~2.

    This is the scatter.c-style check from Bernstein & Khushalani. It is a
    regression guard for the obs_cov_yy indexing bug, which set the Dec variance
    to the RA/Dec off-diagonal term and so under-estimated the minor-axis
    uncertainty by an order of magnitude. It runs across dynamical regimes
    (main-belt and the hyperbolic interstellar object 3I/ATLAS) so the
    covariance is validated where the orbit geometry differs substantially.
    """
    import spiceypy as spice
    from layup.orbitfit import _orbitfit
    from layup.predict import _predict
    from layup.utilities.data_processing_utilities import (
        LayupObservatory,
        layup_furnish_spiceypy,
    )

    DEG = np.pi / 180.0
    ARCSEC = DEG / 3600.0
    OBSCODE = obscode
    N_TRIALS = 100
    SIGMA = 1.0 * ARCSEC  # matches the fitter's default per-observation uncertainty

    layup_furnish_spiceypy(None)
    observatory = LayupObservatory(cache_dir=None)

    obs = CSVDataReader(
        get_test_filepath(obs_filename),
        "csv",
        primary_id_column_name="provID",
    ).read_rows()
    # The et column and observer state depend only on obsTime/station, so they
    # are invariant under the RA/Dec perturbation and can be computed once.
    base = _prep_for_fit(obs, observatory)

    def fit(data):
        return np.atleast_1d(_orbitfit(data, cache_dir=None, primary_id_column_name="provID"))

    # Nominal fit + prediction at the fit epoch.
    fit0 = fit(base)
    assert fit0["flag"][0] == 0, "nominal fit did not converge"
    t_pred = fit0["epochMJD_TDB"][0] + 2400000.5  # MJD TDB -> JD TDB
    et = spice.str2et(f"jd {t_pred} tdb")
    obs_state = np.atleast_1d(
        observatory.obscodes_to_barycentric(np.array([(OBSCODE, et)], dtype=[("stn", "<U10"), ("et", "<f8")]))
    )

    def predict_at(fit_row):
        return _predict(fit_row, obs_state, [t_pred], None, "provID")

    pred0 = predict_at(fit0)
    ra0, dec0 = pred0["ra_deg"][0], pred0["dec_deg"][0]
    cos_dec0 = np.cos(dec0 * DEG)
    # Predicted 2x2 tangent-plane covariance (radians^2), in the (RA*cos(dec), Dec) basis.
    cov_pred = np.array(
        [
            [pred0["obs_cov_xx"][0], pred0["obs_cov_xy"][0]],
            [pred0["obs_cov_xy"][0], pred0["obs_cov_yy"][0]],
        ]
    )
    # The bug made the Dec variance equal to the off-diagonal term.
    assert cov_pred[1, 1] != cov_pred[0, 1], "obs_cov_yy must not equal obs_cov_xy"
    cov_inv = np.linalg.inv(cov_pred)

    # Monte-Carlo: perturb each observation by SIGMA on the sky (RA scaled by
    # 1/cos(dec) so the great-circle displacement is SIGMA), refit, re-predict.
    rng = np.random.default_rng(12345)
    cos_dec_obs = np.cos(base["dec"] * DEG)
    offsets = []
    mahalanobis = []
    for _ in range(N_TRIALS):
        trial = base.copy()
        trial["dec"] = base["dec"] + rng.normal(0.0, SIGMA, len(base)) / DEG
        trial["ra"] = base["ra"] + rng.normal(0.0, SIGMA, len(base)) / DEG / cos_dec_obs
        trial_fit = fit(trial)
        if trial_fit["flag"][0] != 0:
            continue
        pred = predict_at(trial_fit)
        d_a = (pred["ra_deg"][0] - ra0) * DEG * cos_dec0  # tangent-plane east (rad)
        d_d = (pred["dec_deg"][0] - dec0) * DEG  # tangent-plane north (rad)
        offsets.append((d_a, d_d))
        mahalanobis.append(np.array([d_a, d_d]) @ cov_inv @ np.array([d_a, d_d]))

    mahalanobis = np.array(mahalanobis)
    offsets = np.array(offsets)
    assert len(mahalanobis) > 0.9 * N_TRIALS, "too many Monte-Carlo refits failed"

    # Squared Mahalanobis distances ~ chi^2(2); mean ~ 2. Generous band tolerates
    # Monte-Carlo noise and mild fit non-linearity, but rejects the ~10x
    # under-estimate produced by the obs_cov_yy bug (which gave mean ~13).
    assert 1.0 < mahalanobis.mean() < 4.0, f"mean Mahalanobis^2 = {mahalanobis.mean():.2f}"

    # The empirical per-axis scatter must agree with the predicted covariance to
    # within a factor of ~2 (the bug under-predicted the Dec variance by ~10x).
    cov_emp = np.cov(offsets.T)
    for k, axis in enumerate(("RA", "Dec")):
        ratio = cov_emp[k, k] / cov_pred[k, k]
        assert 0.4 < ratio < 2.5, f"{axis} variance ratio empirical/predicted = {ratio:.2f}"
