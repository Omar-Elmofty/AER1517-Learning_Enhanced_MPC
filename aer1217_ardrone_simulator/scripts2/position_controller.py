#!/usr/bin/env python2

"""Class for writing position controller."""

from __future__ import division, print_function, absolute_import

# Import ROS libraries
import roslib
import rospy
import numpy as np
import math 
import time

# Import class that computes the desired positions
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import TransformStamped, Twist
from tf.transformations import euler_from_quaternion


class PositionController(object):
    """ROS interface for controlling the Parrot ARDrone in the Vicon Lab."""
    # write code here for position controller
    def __init__ (self):

    	#Define PID parameters for x and y directions	
		self.Kp_x = 1
		self.Kp_y = 1	
		self.Kv_x = 8 #16	
		self.Kv_y = 8 #16
			
		#Define PID parameters for the z direction
		self.Kp_z = 10
		self.Kv_z = 12
		#Define PID params  for psi angle
		self.Kp_psi = 10
		self.Kv_psi = 4

		self.g = 9.81


    def pos_cont(self,actual_pos, desired_pos, actual_vel, desired_vel):
		"""Function that implements a PID controller for xyz positions
		of an ardrone

		Args:
			actual_pos: TransformStamped msg with actual x,y,z,yaw
			desired_pos: TransformStamped msg with desired x,y,z,yaw
			actual_vel: 4D numpy array with actual rates in x,y,z,yaw
			desired_vel: 4D numpy array with desired rates in x,y,z,yaw
			

		Returns:
			phi_c: command pitch angle
			theta_c: command roll angle
			z_acc: command z acceleration

		"""
		#Retrieve actual and desired positions
		xdes = desired_pos.transform.translation.x
		ydes = desired_pos.transform.translation.y
		zdes = desired_pos.transform.translation.z
		psi_des = desired_pos.transform.rotation.z

		xdes_dot = desired_vel[0]
		ydes_dot = desired_vel[1]
		zdes_dot = desired_vel[2]
		psi_des_dot = desired_vel[3]


		xact = actual_pos.transform.translation.x
		yact = actual_pos.transform.translation.y
		zact = actual_pos.transform.translation.z

		x_dot = actual_vel[0]
		y_dot = actual_vel[1]
		z_dot = actual_vel[2]
		psi_dot = actual_vel[3]

		#convert rotation from quaternion to euler angles
		rotation = (actual_pos.transform.rotation.x,
		    actual_pos.transform.rotation.y,
		    actual_pos.transform.rotation.z,
		    actual_pos.transform.rotation.w)
		euler_angle = euler_from_quaternion(rotation)

		phi_act = euler_angle[0]
		theta_act = euler_angle[1]
		psi_act = euler_angle[2]



		#Calculate accelerations using PID controller
		x_acc = self.Kp_x*(xdes - xact) + self.Kv_x*(xdes_dot - x_dot) 
		y_acc = self.Kp_y*(ydes - yact) + self.Kv_y*(ydes_dot - y_dot)
		z_d = self.Kp_z*(zdes - zact) #+ self.Kv_z*(zdes_dot - z_dot)

		#Ensure delta psi doesn't exceed pi
		delta_psi = psi_des - psi_act
		

		if delta_psi > np.pi:
			delta_psi = delta_psi - 2 * np.pi 
		elif delta_psi < - np.pi:
			delta_psi = delta_psi + 2 * np.pi  	

		#PID controller for psi
		psi_d = self.Kp_psi * delta_psi #+ self.Kv_psi * (psi_des_dot - psi_dot)

		#Clip z and psi accelerations to avoid aggressive maneuvers 
		#z_acc = np.clip(z_acc, -0.5, 0.5)
		#psi_acc = np.clip(psi_acc, -1000.0, 1000.0)

		#calculate the normalized force
		f = (self.g)/(np.cos(theta_act)*np.cos(phi_act))

		#Calculate commanded phi
		y_acc = np.clip(y_acc,-f,f)
		phi_c = np.arcsin(-y_acc/float(f))
		
		#Calculate commanded theta
		x_acc = np.clip(x_acc,-f*np.cos(phi_c), f*np.cos(phi_c))
		theta_c = np.arcsin(x_acc/float(f*np.cos(phi_c)))

		#Adjust the command pitch and roll for yaw
		phi_cmd = phi_c*np.cos(psi_act) + theta_c*np.sin(psi_act)
		theta_cmd = - phi_c * np.sin(psi_act) + theta_c * np.cos(psi_act)


		return phi_cmd, theta_cmd, z_d, psi_d
