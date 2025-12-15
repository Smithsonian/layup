import argparse
import os

import pytest
from numpy.testing import assert_allclose, assert_equal
from layup.comet import _remove_spc, _assist_integrate, _direction_of_integration, comet_cli
from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader
import pandas as pd
import assist
from sorcha.ephemeris.simulation_setup import create_assist_ephemeris, generate_simulations
from layup.utilities.layup_configs import (
    LayupConfigs,
    AuxiliaryConfigs,
)
from argparse import Namespace

from layup.utilities.data_processing_utilities import layup_furnish_spiceypy

def test_remove_spc(tmpdir):
    'Read in data, pass it through _remove_spc and compare to expected data'
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
    assert data['ObjID'].all() == expected['ObjID'].all()

@pytest.mark.parametrize(
    "time_step, index, include_assist",
    [
        (0.5, 0, True),
        (1, 23, True),
        (2, 56, True),
        (2.5, 78, True),
        (-1, 101, True),
        (0, 203, True),
        (10, 45, False),
        (0.01, 324, False),
        (20, 80, False),
        (-5, 4, False),
        (-10, 405, False),
        (-200, -1, False),
    ],
)
def test_assist_integrate(tmpdir, cache, index, time_step, include_assist):
    'Test the function _assist_integrate to make sure it integrates forward or backwards in time as expected.'
    os.chdir(tmpdir)

    data = CSVDataReader(
    get_test_filepath("COM_LPCs_NEW.csv"), "csv", primary_id_column_name="ObjID"
    ).read_rows()

    class FakeArgs:
        primary_id_column_name = "ObjID"
        ar_data_file_path = None

    args = Namespace(input='IGNORE_ME', ar_data_file_path=None, config=None, chunk=10000, force=True, i='csv', o='cometed_output', n=1, primary_id_column_name='ObjID')
    configs = LayupConfigs()
    aux = LayupConfigs.auxiliary

    ephem, Msun, Mtot = create_assist_ephemeris(args, configs.auxiliary)
    
    # Convert to pandas to use 
    cols = data.dtype.names
    orbit_df = pd.DataFrame(data, columns=cols, index=data['ObjID'])
    sim_dict = generate_simulations(ephem, Msun, Mtot, orbit_df, args)
    sim = sim_dict[data[index]['ObjID']]['sim']
    ex = sim_dict[data[index]['ObjID']]['ex']

    if include_assist == False:
        sim = assist.simulation_convert_to_rebound(sim, ephem)

    oi, of, sim = _assist_integrate(sim, ephem, ex, time_step, include_assist=include_assist)

    print()

def test_direction_of_integration():
    print()

def test_apply_comet():
    print()
    
def test_comet_output():
    print()