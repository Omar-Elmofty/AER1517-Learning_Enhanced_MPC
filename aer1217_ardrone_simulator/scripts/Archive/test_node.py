#!/usr/bin/env python2

"""ROS Node for publishing desired positions."""

from __future__ import division, print_function, absolute_import

# Import ROS libraries
import roslib
import rospy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Import class that computes the desired positions
# from aer1217_ardrone_simulator import PositionGenerator

from aer1217_ardrone_simulator.msg import FlatState
from std_msgs.msg import Int16
from narrow_window import Window
from helper_functions import polynomial_interp, nearby_nodes, get_total_cost, truncate, min_snap_trajectory
import matplotlib.animation as animation
from matplotlib import style


if __name__ == '__main__':

    state_init = np.array([0,0,0,0,0,0,0,0,0])
    state_final = np.array([1,1,0,0,0,0,0,0,0])
    int_points = [np.array([0,1,0])]
    mpc_dt = 1/10.0
    s, traj = min_snap_trajectory(state_init, state_final, int_points, mpc_dt)
    plt.figure()
    plt.plot(traj[0,:], traj[1,:])
    plt.figure()
    plt.title('x-pos')
    plt.plot(traj[0,:])
    plt.figure()
    plt.title('y-pos')
    plt.plot(traj[1,:])
    plt.figure()
    plt.title('x-vel')
    plt.plot(traj[3,:])
    plt.figure()
    plt.title('y-vel')
    plt.plot(traj[4,:])
    plt.figure()
    plt.title('x-acc')
    plt.plot(traj[6,:])
    plt.figure()
    plt.title('y-acc')
    plt.plot(traj[7,:])


    plt.show()
