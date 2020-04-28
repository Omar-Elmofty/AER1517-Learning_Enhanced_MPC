#!/usr/bin/env python2

"""ROS Node for simulating the interfaces with the equipment in the vicon lab.

2019-01-23 -- Jacky Liao: added takeoff and land
2017-01-31 -- Rikky Duivenvoorden 

This ROS node subscribes to the following topics:
/cmd_vel_RHC
/gazebo_state

This ROS node publishes to the following topics:
/vicon/ARDroneCarre/ARDroneCarre
/vel_prop

"""

from __future__ import division, print_function, absolute_import

# Import ROS libraries
import roslib
import rospy
import numpy as np

# Import class that computes the desired positions
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import TransformStamped, Twist
from aer1217_ardrone_simulator.msg import MotorCommands
from aer1217_ardrone_simulator.msg import GazeboState
from std_msgs.msg import Empty

class ROSLabInterface(object):
    """ROS interface for emulating the lab equipment."""
    
    def __init__(self):
        """Initialize the ROSLabInterface class."""
        
        # Publishers
        self.pub_vel_prop = rospy.Publisher('/aer1217_ardrone/vel_prop', 
                                            MotorCommands, queue_size=300)
        
        self.model_name = 'ARDroneCarre'
        
        self.pub_vicon_data = rospy.Publisher('/vicon/{0}/{0}'.format(
                                              self.model_name),
                                              TransformStamped, queue_size=30)

        
        # Subscribers
        self.sub_gazebo_pose = rospy.Subscriber('/aer1217_ardrone/gazebo_state', 
                                                GazeboState,
                                                self.update_quadrotor_state)
        
        self.sub_cmd_vel = rospy.Subscriber('cmd_vel_RHC', 
                                            Twist,
                                            self.update_offboard_command)
        
        
        # Initialize messages for publishing
        self.vel_prop_msg = MotorCommands()
        self.quadrotor_state = TransformStamped()
        
        # Run the onboard controller at 200 Hz
        self.onboard_loop_frequency = 200.
        
        # Create an onboard controller for calculation of the motor commands
        self.onboard_controller = ARDroneOnboardController()
        
        # Run this ROS node at the onboard loop frequency
        self.pub_prop_vel = rospy.Timer(rospy.Duration(1. / 
            self.onboard_loop_frequency), self.update_motor_speeds)
        
        # Keep time for differentiation and integration within the controller
        self.old_time = rospy.get_time()
        
    def update_motor_speeds(self, event):
        """Determine the motor speeds and and publishes these."""
        
        # Determine the time step for differentiation and integration
        current_time = rospy.get_time()
        dt = current_time - self.old_time
        
        # Get the motor desired speeds from the onboard controller
        motor_control = self.onboard_controller.get_control_input(dt)
        [front_left, front_right, rear_left, rear_right] = motor_control
        
        # Set the motor_cmd with the controller values
        self.vel_prop_msg.motor_cmd = [front_left, front_right, rear_left, rear_right]

        # Publish the motor commands for the ardrone plugin
        self.pub_vel_prop.publish(self.vel_prop_msg)
        
        # Set the old time to the current for future time step calculations
        self.old_time = current_time
    
    def update_quadrotor_state(self, gazebo_state_msg):
        """Get the gazebo position and attitude for use by controller."""
        
        # Update the quadrotor state with translation and rotation
        (self.quadrotor_state.transform.translation.x,
        self.quadrotor_state.transform.translation.y,
        self.quadrotor_state.transform.translation.z) = (gazebo_state_msg.position[0],
             gazebo_state_msg.position[1],
             gazebo_state_msg.position[2])
        
        (self.quadrotor_state.transform.rotation.x,
        self.quadrotor_state.transform.rotation.y,
        self.quadrotor_state.transform.rotation.z,
        self.quadrotor_state.transform.rotation.w) = (gazebo_state_msg.quaternion[0],
             gazebo_state_msg.quaternion[1],
             gazebo_state_msg.quaternion[2],
             gazebo_state_msg.quaternion[3])
        
        self.quadrotor_state.header.stamp = rospy.Time.now()
        
        # Update the internal state of the onboard controller
        self.onboard_controller.internal_state = self.quadrotor_state
        
        # Republish the state data as a vicon message
        self.pub_vicon_data.publish(self.quadrotor_state)
        
    
    def update_offboard_command(self, cmd_vel_msg):
        """Get the roll-pitch-yaw_rate-climb_rate commands."""
        self.onboard_controller.desired_roll = cmd_vel_msg.linear.x
        self.onboard_controller.desired_pitch = cmd_vel_msg.linear.y
        self.onboard_controller.desired_yaw_rate = cmd_vel_msg.angular.z
        self.onboard_controller.desired_climb_rate = cmd_vel_msg.linear.z


class ARDroneOnboardController(object):
    """Emulation of the Parrot ARDrone 2.0 onboard controller."""
        
    def __init__(self):
        """Initialize the ROSLabInterface class."""

	###
	#subscribers for takeoff and landing 

	self.sub_land = rospy.Subscriber('/ardrone/land', 
                                            Empty,
                                            self.land)

	self.sub_takeoff = rospy.Subscriber('/ardrone/takeoff', 
                                            Empty,
                                            self.takeoff)
	###

        
        # Desired roll and pitch in radians, yaw rate in radians per second, and
        # climb rate in meters per second
        self.desired_roll = 0
        self.desired_pitch = 0
        self.desired_yaw_rate = 0
        self.desired_climb_rate = 0
        
        # Unit commands for controlling a quadrotor. Commands are in following 
        # order: [front_left, front_right, rear_left, rear_right]
        self.unit_hover_cmd = np.array([1., 1., 1., 1.])
        self.unit_roll_cmd = np.array([1., -1., 1., -1.])
        self.unit_pitch_cmd = np.array([-1., -1., 1., 1.])
        self.unit_yaw_cmd = np.array([-1., 1., 1., -1.])
        
        # Onboard controller gains
        self.roll_Pgain = 2.5
        self.roll_Igain = 1.0
        self.roll_Dgain = 1.0
        
        self.pitch_Pgain = 2.5
        self.pitch_Igain = 1.0
        self.pitch_Dgain = 1.0
        
        self.yaw_rate_Pgain = 2.0
        self.yaw_rate_Igain = 0.0
        self.yaw_rate_Dgain = 0.0
        
        self.climb_Pgain = 35.0
        self.climb_Igain = 10.0
        self.climb_Dgain = 0.0
        
        # Motor speed which produces hover
        self.hover_speed = 70.0
        
        # Internal state
        self.internal_state = TransformStamped()
        self.internal_roll_error = 0.0
        self.internal_pitch_error = 0.0
        self.internal_yaw_error = 0.0
        self.internal_climb_error = 0.0
        
        self.internal_old_roll_error = 0.0
        self.internal_old_pitch_error = 0.0
        self.internal_old_yaw_rate_error = 0.0
        self.internal_old_climb_rate_error = 0.0
        
        self.yaw_old = 0.0
        self.z_old = 0.0
        
        # Motor speed envelope
        self.motor_cmd_min = 10
        self.motor_cmd_max = 100

    ####
    # subscribers to set hover speeds on land and takeoff               
    def land(self, msg):
		self.hover_speed = 0.0

    def takeoff(self, msg):
		self.hover_speed = 70.0                                

    ####

    def get_control_input(self, dt):
        """Simulates the onboard controller.
        
        Parameters
        ----------
        dt: float
            Change in time (seconds) used for differentiation and intergration 
            purposes.
        
        Returns
        -------
        motor_control: 4darray
            The vector containing the motor commands for the quadrotor.
            [front_left, front_right, rear_left, rear_right]
        """
        
        desired_z = 3.0
        actual_z = self.internal_state.transform.translation.z
        
        # Convert to quaternion object for use by euler_from_quaternion()
        quaternion = np.array([self.internal_state.transform.rotation.x,
                              self.internal_state.transform.rotation.y,
                              self.internal_state.transform.rotation.z,
                              self.internal_state.transform.rotation.w])
        
        # Determine the euler angles
        euler = euler_from_quaternion(quaternion)
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]
        
        # Corrections for yaw wrap around
        if (not np.sign(yaw) == np.sign(self.yaw_old) and 
            np.abs(np.abs(yaw) - np.pi) < 0.2):
            # Add or subtract 2 pi depending on crossing direction
            self.yaw_old = self.yaw_old + np.sign(yaw) * 2 * np.pi        
        
        
        # Determine current yaw rate for yaw rate controller
        yaw_rate = (yaw - self.yaw_old) / dt
        
        # After use, update the old yaw value with the current yaw
        self.yaw_old = yaw
        
        # Determine current climb rate
        climb_rate = (actual_z - self.z_old) / dt
        
        # After use, update the old altitude with the current altitude
        self.z_old = actual_z
        
        # Find the errors between desired and actual signals
        err_roll = self.desired_roll - roll
        err_pitch = self.desired_pitch - pitch
        err_yaw_rate = self.desired_yaw_rate - yaw_rate
        err_climb_rate = self.desired_climb_rate - climb_rate
        
        # Set the hover motor speed
        motor_control = self.hover_speed * self.unit_hover_cmd
        
        # Roll controller
        motor_control += self.roll_Pgain * self.unit_roll_cmd * err_roll
        
        # Pitch controller
        motor_control += self.roll_Pgain * self.unit_pitch_cmd * err_pitch
        
        # Yaw rate controller (assuming small angles)
        motor_control += self.yaw_rate_Pgain * self.unit_yaw_cmd * err_yaw_rate
        
        # Climb rate controller
        motor_control += ((self.climb_Pgain * err_climb_rate +
                          self.climb_Igain * self.internal_climb_error +
                          self.climb_Dgain / dt * (err_climb_rate - 
                          self.internal_old_climb_rate_error)) * 
                          self.unit_hover_cmd)
        
        # Update the cumulative errors for integration
        self.internal_roll_error += err_roll * dt
        self.internal_pitch_error += err_pitch * dt
        self.internal_yaw_error += err_yaw_rate * dt
        self.internal_climb_error += err_climb_rate * dt
        
        # Update old error with current errors for differentiation
        self.internal_old_roll_error = err_roll
        self.internal_old_pitch_error = err_pitch
        self.internal_old_yaw_rate_error = err_yaw_rate
        self.internal_old_climb_rate_error = err_climb_rate
        
        
        # Return the minimum of the 
        return np.clip(motor_control, self.motor_cmd_min, self.motor_cmd_max)
        

if __name__ == '__main__':
    rospy.init_node('lab_interface')
    ROSLabInterface()
    rospy.spin()
