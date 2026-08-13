import os
import numpy as np
import layup.utilities.herget_iod as herget

from layup.utilities.layup_configs import LayupConfigs
from sorcha.ephemeris.simulation_setup import create_assist_ephemeris

def test_find_mag_to_adjust():
    '''Testing that the formula for finding the closest point on a line works by using a trivial case'''

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
    '''Start with an object moving away from target and check it changes direction of velocity
    Also check end position ends up closer to the target'''

    pos = np.array([50, 0, 0])
    vel = np.array([-1e-5, -1e-5, -1e-5])
    target = np.array([55, 5, 5]) # Function should attempt to add a positive velocity in any cartesian direction to get to this position

    t1 = 0
    tn = 300

    [vx1, vy1, vz1], [*pos_new, vxn, vyn, vzn] = herget.find_new_vel_with_universal_kepler(t1, tn, *pos, *vel, *target)


    assert ([vx1, vy1, vz1] > vel).all()
    assert (np.sqrt(sum((target - pos_new)**2)) < np.sqrt(sum((target - pos)**2))).all() # Check the new end-position is closer to the target

    # Run again to check it continues to converge
    [vx1, vy1, vz1], [*pos_new_rerun, vxn, vyn, vzn] = herget.find_new_vel_with_universal_kepler(t1, tn, *pos, vx1, vy1, vz1, *target)
    
    assert (np.sqrt(sum((target - pos_new_rerun)**2)) < np.sqrt(sum((target - pos_new)**2))).all()

def test_find_new_vel():
    '''Same as above but for using assist variational particles'''

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
    ephem, _, _ = create_assist_ephemeris(args, aux)

    pos = np.array([50, 0, 0])
    vel = np.array([-1e-5, -1e-5, -1e-5])
    target = np.array([55, 5, 5]) # Function should attempt to add a positive velocity in any cartesian direction to get to this position

    t1 = 2406000 
    tn = 2406300 

    vx1, vy1, vz1, vxn, vyn, vzn, pos_new = herget.find_new_vel(ephem, t1, tn, *pos, *vel, *target, change = 'x')
    vx1, vy1, vz1, vxn, vyn, vzn, pos_new = herget.find_new_vel(ephem, t1, tn,  *pos, vx1, vy1, vz1, *target, change = 'y')
    vx1, vy1, vz1, vxn, vyn, vzn, pos_new = herget.find_new_vel(ephem, t1, tn,  *pos, vx1, vy1, vz1, *target, change = 'z')


    assert ([vx1, vy1, vz1] > vel).all()
    assert (np.sqrt(sum((target - pos_new)**2)) < np.sqrt(sum((target - pos)**2))).all() # Check the new end-position is closer to the target

    # Run again to check it continues to converge
    vx1, vy1, vz1, vxn, vyn, vzn, pos_new_rerun = herget.find_new_vel(ephem, t1, tn,  *pos, vx1, vy1, vz1, *target, change = 'x')
    vx1, vy1, vz1, vxn, vyn, vzn, pos_new_rerun = herget.find_new_vel(ephem, t1, tn,  *pos, vx1, vy1, vz1, *target, change = 'y')
    vx1, vy1, vz1, vxn, vyn, vzn, pos_new_rerun = herget.find_new_vel(ephem, t1, tn,  *pos, vx1, vy1, vz1, *target, change = 'z')

    assert (np.sqrt(sum((target - pos_new_rerun)**2)) < np.sqrt(sum((target - pos_new)**2))).all()

def test_find_drho():
    '''Take an object with a known rho value (from JPL), see if this function approaches the right direction'''
    