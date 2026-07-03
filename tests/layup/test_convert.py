import argparse
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from layup.convert import convert, convert_cli
from layup.utilities.data_processing_utilities import has_cov_columns
from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader
from layup.utilities.file_io.HDF5Reader import HDF5DataReader
from layup.utilities.orbit_conversion import (
    covariance_keplerian_xyz,
    universal_cartesian,
    universal_keplerian,
)

# Standard gravitational parameter of the Sun, AU^3 / day^2.
_GM_SUN = 0.00029591220828559


def create_argparse_object():
    parser = argparse.ArgumentParser(description="Convert orbital data formats.")
    parser.add_argument("--ar_data_file_path", type=str, required=False, help="cache directory")
    parser.add_argument(
        "--primary-id-column-name",
        type=str,
        default="ObjID",
        required=False,
        help="Column name for primary ID",
    )

    args = parser.parse_args([])

    return args


def test_convert_round_trip():
    """Convert into all 6 possible output formats and then conver the output back into its original format."""
    # TODO(wbeebe): Add additional test files to test more input formats.
    csv_input_files = ["BCOM.csv", "KEP.csv", "one_cent_orbs.csv", "two_cent_orbs.csv"]
    for csv_input_file in csv_input_files:
        input_csv_reader = CSVDataReader(get_test_filepath(csv_input_file))
        input_data = input_csv_reader.read_rows()
        input_format = input_data[0]["FORMAT"]
        if input_format == "BCOM":
            # TODO(wbeebe): The last row is a hyperbolic orbit in barycentric coordinates.
            # It is removed here to avoid a conversion failure, but we should handle this
            # case in the future, and add this row back into the test
            input_data = input_data[0 : len(input_data) - 1]
            assert len(input_data) == 813
        for output_format in ["BCOM", "BCART", "BCART_EQ", "BKEP", "COM", "CART", "KEP"]:
            # Convert to the output format.
            first_convert_data = convert(input_data, output_format, num_workers=1)
            # Convert back to the original format for round trip checking.
            output_data = convert(first_convert_data, input_format, num_workers=1)

            assert_equal(len(input_data), len(output_data))

            # Test that the columns are the same. Note that column order may not be preserved.
            for column_name in input_data.dtype.names:
                # For non-numeric columns, we can't use assert_allclose, so we use assert_equal.
                if (
                    input_data[column_name].dtype.kind == "S"
                    or input_data[column_name].dtype.kind == "U"
                    or input_data[column_name].dtype.kind == "O"
                ):
                    assert_equal(
                        input_data[column_name],
                        output_data[column_name],
                        err_msg=f"Column {column_name} not equal with dtype {input_data[column_name].dtype} after converting from {csv_input_file} to {output_format} and back",
                    )
                else:
                    # Test that we convert back to our original numeric values within a small tolerance of lost precision.
                    assert_allclose(
                        input_data[column_name],
                        output_data[column_name],
                        err_msg=f"Column {column_name} not equal with dtype {input_data[column_name].dtype} after converting from {csv_input_file} to {output_format} and back",
                    )


@pytest.mark.parametrize(
    "chunk_size, num_workers",
    [
        (10_0000, 1),
    ],
)
def test_convert_round_trip_csv(tmpdir, chunk_size, num_workers):
    """Test that the convert function works for a small CSV file."""
    cli_args = create_argparse_object()
    input_file = get_test_filepath("BCOM.csv")
    input_csv_reader = CSVDataReader(input_file, "csv")
    input_data = input_csv_reader.read_rows()

    # Since the convert CLI outputs to the current working directory, we need to change to our temp directory
    output_file_stem_BCART = "test_output_BCART"
    os.chdir(tmpdir)
    temp_BCART_out_file = os.path.join(tmpdir, f"{output_file_stem_BCART}.csv")
    # Convert our BCOM CV file to a BCART CSV file
    convert_cli(
        input_file,
        output_file_stem_BCART,
        "BCART",
        "csv",
        chunk_size=chunk_size,
        num_workers=num_workers,
        cli_args=cli_args,
    )

    # Verify the conversion produced an output file
    assert os.path.exists(temp_BCART_out_file)

    # Create a new CSV reader to read in our output BCART file
    output_csv_reader = CSVDataReader(temp_BCART_out_file, "csv")
    output_data_BCART = output_csv_reader.read_rows()
    # Verify that the number of rows in the input and output files are the same
    assert_equal(len(input_data), len(output_data_BCART))

    # Now convert back to BCOM so we can verify the round trip conversion.
    output_file_stem_BCOM = "test_output_BCOM"
    temp_BCOM_out_file = os.path.join(tmpdir, f"{output_file_stem_BCOM}.csv")
    convert_cli(
        temp_BCART_out_file,
        output_file_stem_BCOM,
        "BCOM",
        "csv",
        chunk_size=chunk_size,
        num_workers=num_workers,
        cli_args=cli_args,
    )

    # Verify the conversion produced an output file
    assert os.path.exists(temp_BCOM_out_file)
    output_csv_reader = CSVDataReader(temp_BCOM_out_file, "csv")
    output_data_BCOM = output_csv_reader.read_rows()

    # Test that the file has the same number of rows and columns as our input file
    assert_equal(len(input_data), len(output_data_BCOM))
    assert_equal(set(input_data.dtype.names), set(output_data_BCOM.dtype.names))

    # Test that the columns have equivalent values, note that column order may have changed.
    for column_name in input_data.dtype.names:
        # For non-numeric columns, we can't use assert_allclose, so we use assert_equal.
        if (
            input_data[column_name].dtype.kind == "S"
            or input_data[column_name].dtype.kind == "U"
            or input_data[column_name].dtype.kind == "O"
        ):
            assert_equal(
                input_data[column_name],
                output_data_BCOM[column_name],
                err_msg=f"Column {column_name} not equal with dtype {input_data[column_name].dtype}",
            )
        else:
            # Test that we convert back to our original numeric values within a small tolerance of lost precision.
            assert_allclose(
                input_data[column_name],
                output_data_BCOM[column_name],
                err_msg=f"Column {column_name} not equal with dtype {input_data[column_name].dtype}",
            )


@pytest.mark.parametrize(
    "chunk_size, num_workers, output_format",
    [
        (10_0000, 1, "BKEP"),
        (10_0000, 1, "BCOM"),
        (10_0000, 1, "COM"),
        (10_0000, 1, "KEP"),
        (10_0000, 1, "BCART"),
        (10_0000, 1, "CART"),
    ],
)
def test_convert_BCART_EQ_csv_with_covariance(tmpdir, chunk_size, num_workers, output_format):
    """Test that the convert function works for a small CSV file."""
    cli_args = create_argparse_object()
    cli_args.primary_id_column_name = "provID"
    input_file = get_test_filepath("test_convert_BCART_EQ.csv")
    input_csv_reader = CSVDataReader(input_file, "csv", primary_id_column_name="provID")
    input_data = input_csv_reader.read_rows()

    # Since the convert CLI outputs to the current working directory, we need to change to our temp directory
    output_file_stem = f"test_output_{output_format}"
    os.chdir(tmpdir)
    temp_out_file = os.path.join(tmpdir, f"{output_file_stem}.csv")
    # Convert our BCART_EQ CSV file to a different format CSV file
    convert_cli(
        input_file,
        output_file_stem,
        output_format,
        "csv",
        chunk_size=chunk_size,
        num_workers=num_workers,
        cli_args=cli_args,
    )

    # Verify the conversion produced an output file
    assert os.path.exists(temp_out_file)

    # Create a new CSV reader to read in our output file
    output_csv_reader = CSVDataReader(temp_out_file, "csv", primary_id_column_name="provID")
    output_data = output_csv_reader.read_rows()
    # Verify that the number of rows in the input and output files are the same
    assert_equal(len(input_data), len(output_data))

    # Now convert that output file back to BCART so we can verify the round trip conversion.
    output_file_stem_BCART_EQ = "final_output_BCART_EQ"
    temp_BCART_EQ_out_file = os.path.join(tmpdir, f"{output_file_stem_BCART_EQ}.csv")
    convert_cli(
        temp_out_file,
        output_file_stem_BCART_EQ,
        "BCART_EQ",
        "csv",
        chunk_size=chunk_size,
        num_workers=num_workers,
        cli_args=cli_args,
    )

    # Verify the conversion produced an output file
    assert os.path.exists(temp_BCART_EQ_out_file)
    output_csv_reader = CSVDataReader(temp_BCART_EQ_out_file, "csv", primary_id_column_name="provID")
    output_data_BCART_EQ = output_csv_reader.read_rows()

    # Test that the file has the same number of rows and columns as our input file
    assert_equal(len(input_data), len(output_data_BCART_EQ))
    # assert_equal(set(input_data.dtype.names), set(output_data_BCOM.dtype.names))

    # Test that the columns have equivalent values, note that column order may have changed.
    assert has_cov_columns(input_data)
    assert has_cov_columns(output_data_BCART_EQ)
    for column_name in output_data_BCART_EQ.dtype.names:
        # For non-numeric columns, we can't use assert_allclose, so we use assert_equal.
        if (
            input_data[column_name].dtype.kind == "S"
            or input_data[column_name].dtype.kind == "U"
            or input_data[column_name].dtype.kind == "O"
        ):
            assert_equal(
                input_data[column_name],
                output_data_BCART_EQ[column_name],
                err_msg=f"Column {column_name} not equal with dtype {input_data[column_name].dtype}",
            )
        elif column_name:
            # Test that we convert back to our original numeric values within a small tolerance of lost precision.
            assert_allclose(
                input_data[column_name],
                output_data_BCART_EQ[column_name],
                err_msg=f"Column {column_name} not equal with dtype {input_data[column_name].dtype}",
            )


@pytest.mark.parametrize(
    "chunk_size, num_workers",
    [
        (10_0000, 1),
    ],
)
def test_convert_round_trip_hdf5(tmpdir, chunk_size, num_workers):
    # Test that the convert function works for a small HDF5 file.
    cli_args = create_argparse_object()
    input_file_BCOM = get_test_filepath("BCOM.h5")
    input_hdf5_reader = HDF5DataReader(input_file_BCOM)
    input_data_BCOM = input_hdf5_reader.read_rows()

    # Since the convert CLI outputs to the current working directory, we need to change to our temp directory
    output_file_stem_BCART = "test_output_BCART"
    os.chdir(tmpdir)
    temp_out_file_BCART = os.path.join(tmpdir, f"{output_file_stem_BCART}.h5")

    # Convert our BCOM HDF5 file to a BCART HDF5 file
    convert_cli(
        input_file_BCOM,
        output_file_stem_BCART,
        "BCART",
        "hdf5",
        chunk_size=chunk_size,
        num_workers=num_workers,
        cli_args=cli_args,
    )

    # Verify the conversion produced an output file
    assert os.path.exists(temp_out_file_BCART)
    output_hdf5_reader = HDF5DataReader(temp_out_file_BCART)
    output_data_BCART = output_hdf5_reader.read_rows()
    assert_equal(len(input_data_BCOM), len(output_data_BCART))

    # Convert our output BCART file back to BCOM so we can verify the round trip conversion.
    output_file_stem_BCOM = "test_output_BCOM"
    temp_BCOM_out_file = os.path.join(tmpdir, f"{output_file_stem_BCOM}.h5")
    convert_cli(
        temp_out_file_BCART,
        output_file_stem_BCOM,
        "BCOM",
        "hdf5",
        chunk_size=chunk_size,
        num_workers=num_workers,
        cli_args=cli_args,
    )

    # Verify the conversion produced an output file
    assert os.path.exists(temp_BCOM_out_file)
    output_hdf5_reader = HDF5DataReader(temp_BCOM_out_file)
    output_data_BCOM = output_hdf5_reader.read_rows()

    # Test that the file has the same number of rows and columns as our input file
    assert_equal(len(input_data_BCOM), len(output_data_BCOM))
    assert_equal(set(input_data_BCOM.dtype.names), set(output_data_BCOM.dtype.names))

    # Test that the columns have equivalent values, note that column order may have changed.
    for column_name in input_data_BCOM.dtype.names:
        # For non-numeric columns, we can't use assert_allclose, so we use assert_equal.
        if (
            input_data_BCOM[column_name].dtype.kind == "S"
            or input_data_BCOM[column_name].dtype.kind == "U"
            or input_data_BCOM[column_name].dtype.kind == "O"
        ):
            assert_equal(
                input_data_BCOM[column_name],
                output_data_BCOM[column_name],
                err_msg=f"Column {column_name} not equal with dtype {input_data_BCOM[column_name].dtype}",
            )
        else:
            # Test that we convert back to our original numeric values within a small tolerance of lost precision.
            assert_allclose(
                input_data_BCOM[column_name],
                output_data_BCOM[column_name],
                err_msg=f"Column {column_name} not equal with dtype {input_data_BCOM[column_name].dtype}",
            )


def test_keplerian_covariance_finite_for_hyperbolic_orbit():
    """Regression for #288: converting a hyperbolic orbit (e>1, a<0) to KEP/BKEP
    must produce a finite covariance.

    universal_keplerian computed the mean motion as sqrt(mu / a**3); for a < 0
    that is the square root of a negative number -> NaN, which poisoned the KEP
    covariance Jacobian and produced an all-NaN covariance matrix.  The mean
    motion is now sqrt(mu / |a|**3), real for hyperbolic orbits.
    """
    epoch = 60000.0
    # Build a hyperbolic Cartesian state from elements (q = 1.2 AU, e = 1.4).
    state = np.asarray(
        universal_cartesian(_GM_SUN, 1.2, 1.4, 0.3, 0.5, 0.8, epoch - 30.0, epoch),
        dtype=float,
    )
    a, e = (float(v) for v in universal_keplerian(_GM_SUN, *state, epoch)[:2])
    assert a < 0.0 and e > 1.0  # premise: genuinely hyperbolic

    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6)) * 1e-5
    cov_xyz = A @ A.T + np.eye(6) * 1e-11  # a symmetric positive-definite Cartesian covariance
    cov_kep = np.asarray(covariance_keplerian_xyz(_GM_SUN, *state, epoch, cov_xyz))
    assert np.all(np.isfinite(cov_kep)), "hyperbolic KEP covariance contains NaN/inf"


def test_convert_covariance_angle_units():
    """Covariance of degree-valued angle elements must be in degrees^2.

    convert() reports the angular elements (inc, node, argPeri, ma) in degrees.
    Their covariance must be in degrees^2 to match. Regression test for a bug
    where the angle *values* were converted radians->degrees but their
    covariance was left in radians^2, making the reported angular uncertainties
    too small by 180/pi (their variances by (180/pi)^2 ~ 3283x). The existing
    round-trip tests miss this because the inverse conversion undoes both the
    value and the covariance scaling; here we check the forward covariance
    against a finite-difference Jacobian of the conversion.
    """
    data = CSVDataReader(
        get_test_filepath("test_convert_BCART_EQ.csv"), "csv", primary_id_column_name="provID"
    ).read_rows()
    data = np.atleast_1d(data)
    state_cols = ["x", "y", "z", "xdot", "ydot", "zdot"]
    s0 = np.array([data[c][0] for c in state_cols])
    cov_bcart = np.array([[data[f"cov_{i}_{j}"][0] for j in range(6)] for i in range(6)])

    for fmt, elem_cols in [
        ("KEP", ["a", "e", "inc", "node", "argPeri", "ma"]),
        ("COM", ["q", "e", "inc", "node", "argPeri", "t_p_MJD_TDB"]),
    ]:
        conv = convert(data, fmt, num_workers=1, primary_id_column_name="provID")
        cov_analytic = np.array([[conv[f"cov_{i}_{j}"][0] for j in range(6)] for i in range(6)])

        # Finite-difference Jacobian d(format element)/d(BCART_EQ state).
        jac = np.zeros((6, 6))
        for k in range(6):
            h = 1e-7 * max(abs(s0[k]), 1e-3 if k < 3 else 1e-5)
            dp = data.copy()
            dp[state_cols[k]][0] = s0[k] + h
            dm = data.copy()
            dm[state_cols[k]][0] = s0[k] - h
            cp = convert(dp, fmt, num_workers=1, primary_id_column_name="provID")
            cm = convert(dm, fmt, num_workers=1, primary_id_column_name="provID")
            for r in range(6):
                jac[r, k] = (cp[elem_cols[r]][0] - cm[elem_cols[r]][0]) / (2 * h)

        cov_fd = jac @ cov_bcart @ jac.T
        rel_err = np.linalg.norm(cov_analytic - cov_fd) / np.linalg.norm(cov_fd)
        assert rel_err < 1e-3, (
            f"{fmt} covariance is inconsistent with the finite-difference Jacobian "
            f"(rel.err {rel_err:.2e}) -- angle covariance likely in the wrong units"
        )


def test_element_order_matches_degree_columns():
    """Guard the hand-synced invariant _scale_degree_cov relies on.

    `_scale_degree_cov` maps each degree column to a covariance row/col via
    `element_order[fmt].index(col)`. If `degree_columns` and `element_order`
    drift apart the lookup either raises or silently scales the wrong row, so
    pin: every degree format is in element_order, each of its degree columns is
    present there, and every element_order entry is a unique 6-vector (so the
    index is a valid 0..5 covariance position). The *numerical* basis match is
    covered by test_convert_covariance_angle_units.
    """
    from layup.convert import degree_columns, element_order

    for fmt, cols in degree_columns.items():
        assert fmt in element_order, f"{fmt} in degree_columns but not element_order"
        order = element_order[fmt]
        assert len(order) == 6, f"element_order[{fmt}] is not a 6-vector: {order}"
        assert len(set(order)) == 6, f"element_order[{fmt}] has duplicates: {order}"
        for col in cols:
            assert col in order, f"degree column {col!r} missing from element_order[{fmt}]"
            assert 0 <= order.index(col) <= 5


def test_convert_vectorized_matches_rowwise():
    """Pin the vectorized path to the per-row reference (``_apply_convert_rowwise``).

    ``convert()`` always uses ``_apply_convert_vectorized`` now, so this calls both
    implementations directly with identical arguments and requires the converted
    state *and* the propagated covariance to match, for every output format. Guards
    the vectorized ``_parse_to_bcart_eq`` / ``_bcart_eq_to_elements`` /
    ``_parse_cov_to_bcart_eq`` and the vmapped covariance transforms against drifting
    from the scalar routines.
    """
    import numpy.lib.recfunctions as rfn

    from layup.convert import (
        ORBIT_FIT_COLS,
        _apply_convert_rowwise,
        _apply_convert_vectorized,
        _create_assist_ephemeris,
        element_order,
        get_output_column_names_and_types,
    )
    from layup.utilities.data_processing_utilities import get_cov_columns
    from layup.utilities.layup_configs import LayupConfigs

    cfg = LayupConfigs()
    ephem, gm_sun, gm_total = _create_assist_ephemeris(cfg.auxiliary, None)
    formats = ["BCART_EQ", "BCART", "CART", "KEP", "COM", "BKEP", "BCOM"]
    degree_cols = {"inc", "node", "argPeri", "ma"}
    cov_names = get_cov_columns()

    def both_paths(data, convert_to, pid="ObjID"):
        has_cov = has_cov_columns(data)
        keep = [(c, d) for c, d in ORBIT_FIT_COLS if c in data.dtype.names]
        req, dts = get_output_column_names_and_types(pid, has_cov, keep)
        output_dtype = [(c, d) for c, d in zip(req[convert_to], dts)]
        args = (data, convert_to, ephem, gm_sun, gm_total, pid, has_cov, output_dtype, keep)
        return _apply_convert_vectorized(*args), _apply_convert_rowwise(*args)

    def check(data, output_formats, with_cov):
        for output_format in output_formats:
            vec, ref = both_paths(data, output_format)
            cols = list(element_order[output_format]) + ["epochMJD_TDB"]
            if with_cov:
                cols += cov_names
            for col in cols:
                a = np.asarray(vec[col], dtype=float)
                b = np.asarray(ref[col], dtype=float)
                if col in degree_cols:
                    # compare modulo 360 so the 0/360 wrap boundary can't spuriously fail
                    assert_allclose(
                        (a - b + 180) % 360 - 180, 0.0, atol=1e-7, err_msg=f"{col} for {output_format}"
                    )
                else:
                    assert_allclose(a, b, rtol=1e-8, atol=1e-8, err_msg=f"{col} for {output_format}")

    def attach_spd_cov(data):
        rng = np.random.default_rng(0)
        covs = np.stack(
            [(lambda A: A @ A.T + np.eye(6) * 1e-9)(rng.standard_normal((6, 6))) for _ in range(len(data))]
        )
        fields = []
        for nm in cov_names:
            _, i, j = nm.split("_")
            fields.append(covs[:, int(i), int(j)].copy())
        return rfn.append_fields(data, cov_names, fields, usemask=False)

    # Values across all formats, including BCOM.csv's hyperbolic final row.
    check(CSVDataReader(get_test_filepath("BCOM.csv")).read_rows(), formats, with_cov=False)

    # Covariance: attach a non-trivial SPD covariance and compare the propagated
    # covariance too. BCART_EQ input exercises the batched output transforms;
    # BCOM input (elliptic subset) exercises the per-row element-input cov parse.
    kep = CSVDataReader(get_test_filepath("KEP.csv")).read_rows()
    bcart_eq = convert(kep, "BCART_EQ", num_workers=1)
    check(attach_spd_cov(bcart_eq), formats, with_cov=True)

    bcom_elliptic = CSVDataReader(get_test_filepath("BCOM.csv")).read_rows()[:-1]
    check(attach_spd_cov(bcom_elliptic), ["KEP", "BCART_EQ", "BCART"], with_cov=True)
