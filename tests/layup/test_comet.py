import argparse
import os

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from layup.comet import _remove_spc, _assist_integrate, _direction_of_integration, _apply_comet, comet_cli
from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader
import pandas as pd
import assist
from sorcha.ephemeris.simulation_setup import create_assist_ephemeris, generate_simulations
from layup.utilities.layup_configs import (
    LayupConfigs,
)
from argparse import Namespace

from layup.utilities.data_processing_utilities import layup_furnish_spiceypy


def test_remove_spc(tmpdir):
    "Read in data, pass it through _remove_spc and compare to expected data"
    os.chdir(tmpdir)
    data = CSVDataReader(
        get_test_filepath("COM_LPCs_NEW.csv"), "csv", primary_id_column_name="ObjID"
    ).read_rows()

    data = _remove_spc(data)
    # Read in expected data
    expected = CSVDataReader(
        get_test_filepath("COM_LPCs_no_spc.csv"), "csv", primary_id_column_name="ObjID"
    ).read_rows()

    # Check that all of the LPCs are present
    assert data["ObjID"].all() == expected["ObjID"].all()


@pytest.mark.parametrize(
    "time_step, index, include_assist",
    [
        (0.5, 0, True),
        (1, 23, True),
        (2, 56, True),
        (2.5, 79, True),
        (-1, 101, True),
        (0, 203, True),
        (10, 45, False),
        (0.01, 324, False),
        (20, 580, False),
        (-5, 4, False),
        (-10, 305, False),
        (-100, -1, False),
    ],
)
def test_assist_integrate(tmpdir, index, time_step, include_assist):
    "Test the function _assist_integrate to make sure it integrates forward or backwards in time as expected."
    os.chdir(tmpdir)

    data = CSVDataReader(
        get_test_filepath("COM_LPCs_NEW.csv"), "csv", primary_id_column_name="ObjID"
    ).read_rows()

    args = Namespace(
        input="IGNORE_ME",
        ar_data_file_path=None,
        config=None,
        chunk=10000,
        force=True,
        i="csv",
        o="cometed_output",
        n=1,
        primary_id_column_name="ObjID",
    )
    aux = LayupConfigs().auxiliary

    ephem, Msun, Mtot = create_assist_ephemeris(args, aux)

    # Convert to pandas to use generate_simulations
    cols = data.dtype.names
    orbit_df = pd.DataFrame(data, columns=cols, index=data["ObjID"])
    sim_dict = generate_simulations(ephem, Msun, Mtot, orbit_df, args)
    sim = sim_dict[data[index]["ObjID"]]["sim"]
    ex = sim_dict[data[index]["ObjID"]]["ex"]

    if include_assist == False:
        sim = assist.simulation_convert_to_rebound(sim, ephem)

    initial = sim.particles[-1].xyz
    sim.dt = time_step

    oi, of, sim = _assist_integrate(sim, ex, time_step, ephem, include_assist=include_assist)
    initial_orbit, final_orbit = oi.a, of.a

    final = sim.particles[-1].xyz

    # Once we find the returned values, compare that to what we get if we manually set up a rebound simulation (if include_assist is False, compare it to an assist simulation instead)
    sim_dict_check = generate_simulations(ephem, Msun, Mtot, orbit_df, args)
    sim_check = sim_dict_check[data[index]["ObjID"]]["sim"]

    initial_check = sim_check.particles[-1].xyz
    sim_check.dt = time_step

    if include_assist == True:
        sim_check = assist.simulation_convert_to_rebound(sim_check, ephem)
        initial_check = sim_check.particles[-1].xyz
        primary_check = sim_check.particles[0]
        initial_orbit_check = sim_check.particles[-1].orbit(primary=primary_check).a
        sim_check.integrate(sim_check.t + time_step)  # Want to integrate to the same time as the function
        primary_check = sim_check.particles[0]

    else:
        primary_check = ephem.get_particle("sun", sim_check.t)
        initial_orbit_check = sim_check.particles[-1].orbit(primary=primary_check).a
        sim_check.integrate(sim_check.t + time_step)
        primary_check = ephem.get_particle("sun", sim_check.t)
    final_check = sim_check.particles[-1].xyz
    final_orbit_check = sim_check.particles[-1].orbit(primary=primary_check).a

    assert_equal(np.array(initial), np.array(initial_check))
    assert_equal(sim.t, sim_check.t)
    assert_allclose(np.array(final), np.array(final_check), rtol=2e-7)
    assert_allclose(
        np.array([initial_orbit, final_orbit]),
        np.array([initial_orbit_check, final_orbit_check]),
        rtol=1e-2
    )


def test_direction_of_integration(tmpdir):
    "Test the function direction_of_integration to make sure it returns the correct direction in time for the 4 different comet positions"
    os.chdir(tmpdir)

    data = CSVDataReader(
        get_test_filepath("comets_various_positions.csv"), "csv", primary_id_column_name="ObjID"
    ).read_rows()

    class FakeCliArgs:
        def __init__(self, g=None):
            self.primary_id_column_name = "ObjID"
            self.n = 1
            self.chunk = 10000
            self.ar_data_file_path = None
            self.force = True
            self.code_format = True

    args = FakeCliArgs()
    aux = LayupConfigs().auxiliary

    ephem, Msun, Mtot = create_assist_ephemeris(args, aux)

    cols = list(data.dtype.names)
    cols.pop(-1)  # remove the check stored in the last column
    orbit_df = pd.DataFrame(data, columns=cols, index=data["ObjID"])

    sim_dict = generate_simulations(ephem, Msun, Mtot, orbit_df, args)
    for comet in range(len(data)):
        sim = sim_dict[data[comet]["ObjID"]]["sim"]
        ex = sim_dict[data[comet]["ObjID"]]["ex"]
        dt, oi, of = _direction_of_integration(sim, ex, 1, ephem, include_assist=True)
        assert dt == data["expected_step"][comet]


def test_apply_comet(tmpdir):
    "Test _apply_comet against LPCs from the CODE Catalogue."
    os.chdir(tmpdir)

    data = CSVDataReader(
        get_test_filepath("code_LPCs.csv"), "csv", primary_id_column_name="ObjID"
    ).read_rows()

    class FakeCliArgs:
        def __init__(self, g=None):
            self.primary_id_column_name = "ObjID"
            self.n = 1
            self.chunk = 10000
            self.ar_data_file_path = None
            self.force = True
            self.code_format = True

    args = FakeCliArgs()
    aux = LayupConfigs().auxiliary
    output = _apply_comet(data, args, aux, primary_id_column_name="ObjID")

    # Read in expected output
    expected = CSVDataReader(
        get_test_filepath("code_LPCs_originals.csv"), "csv", primary_id_column_name="ObjID"
    ).read_rows()

    # Compare inv_ao to the CODE Catalogue for each comet
    for comet in output["ObjID"]:
        i_test = np.where(output["ObjID"] == comet)[0][0]
        i_expected = np.where(expected["ObjID"] == comet)[0][0]
        assert_allclose(
            output[i_test]["inv_ao_CODE"], expected[i_expected]["inv_ao"], atol=100
        )  # It is difficult to test the accuracy of the simulations without a measure of uncertainties
        # so this tolerance is quite high. The simulations give an order of magnitude agreement with the code catalogue.


def test_comet_output(tmpdir):
    "Test that running layup comet as a user would returns the expected output"
    import subprocess
    from pathlib import Path

    os.chdir(tmpdir)

    test_filename = "demo_comet.csv"
    input_file = Path(get_test_filepath(test_filename))
    temp_out_file = f"test_output{input_file.stem}"

    result = subprocess.run(["layup", "comet", str(input_file), "-f", "-o", str(temp_out_file), "-cf"])

    assert result.returncode == 0

    result_file = Path(f"{tmpdir}/{temp_out_file}.csv")
    assert result_file.exists

    # Create a new CSV reader to read in our output file
    output_csv_reader = CSVDataReader(str(result_file), "csv", primary_id_column_name="ObjID")
    output_data = output_csv_reader.read_rows()

    # Read in the known output
    known_output_file = get_test_filepath("demo_comet_expected.csv")
    known_output_csv_reader = CSVDataReader(known_output_file, "csv", primary_id_column_name="ObjID")
    known_data = known_output_csv_reader.read_rows()

    print("assert 1")
    assert np.allclose(output_data["inv_ao_CODE"], known_data["inv_ao_CODE"], rtol=1e-5)
    print("assert 2")
    assert np.allclose(output_data["ao_barycentric"], known_data["ao_barycentric"])
    print("assert 3")
    assert np.allclose(output_data["d_ao"], known_data["d_ao"])
    print("assert 4")
    assert np.allclose(output_data["e_ao"], known_data["e_ao"])
