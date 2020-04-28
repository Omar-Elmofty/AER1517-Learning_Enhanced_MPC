#!/usr/bin/env python2

"""
ROS Node for controlling the ARDrone 2.0 using the ardrone_autonomy package.

This ROS node subscribes to the following topics:
/vicon/ARDroneCarre/ARDroneCarre

This ROS node publishes to the following topics:
/cmd_vel_RHC

"""

from __future__ import division, print_function, absolute_import

# Import ROS libraries
import roslib
import rospy
import numpy as np

# Import class that computes the desired positions
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import TransformStamped, Twist, Vector3
from position_controller import PositionController
from std_msgs.msg import Int16
from aer1217_ardrone_simulator.msg import FlatState
from aer1217_ardrone_simulator.msg import FlatInput
from numpy import sin,cos,tan


class ROSControllerNode(object):
    """ROS interface for controlling the Parrot ARDrone in the Vicon Lab.
    write code here to define node publishers and subscribers
    publish to /cmd_vel topic
    subscribe to /vicon/ARDroneCarre/ARDroneCarre for position and 
    attitude feedback
    """
    def __init__ (self):

		# Publishers
		self.pub_vel_cmd = rospy.Publisher('/cmd_vel_RHC', 
		                            Twist, queue_size=32)


		self.pub_flat_state = rospy.Publisher('/aer1217/flat_state',
						FlatState,queue_size=32)

		#Subscriber
		self.sub_vicon = rospy.Subscriber('/vicon/ARDroneCarre/ARDroneCarre', 
		                                TransformStamped,
		                                self.update_quadrotor_state)

		self.sub_flat_in = rospy.Subscriber('/aer1217/flat_in', 
		                                FlatInput,
		                                self.update_flat_in)
		self.sub_flat_state_op = rospy.Subscriber('/aer1217/flat_state_op', 
		                                FlatState,
		                                self.update_flat_state_op)

		self.sub_desired_pos = rospy.Subscriber('/aer1217/desired_position', 
				                        TransformStamped,
				                        self.update_desired_pos)

		#initialize  variables
		self.actual_pos = TransformStamped()
		self.actual_pos_prev = TransformStamped()
		self.actual_vel = np.array([0,0,0,0])
		self.actual_vel_prev  = np.array([0,0,0,0])
		self.actual_vel_prev2  = np.array([0,0,0,0])
		self.actual_vel_prev3  = np.array([0,0,0,0])
		self.flat_state_msg = FlatState()

		self.desired_pos = TransformStamped()
		self.desired_pos.transform.translation.z = 2 #initialize z to 2
		self.desired_pos_prev = TransformStamped()
		self.desired_vel = np.array([0,0,2,0])

		self.flat_state = list(np.zeros(9))
		self.flat_input = list(np.zeros(4))
		self.flat_state_op = list(np.zeros(9))


		#initialize position controller
		self.pos_controller = PositionController()
        	self.flat_input = np.array([0,0,0,0])

		#publish vel commands
		self.dt = 1/300.0
		self.freq = 300

		#run ros loop		
		rate = rospy.Rate(self.freq)

	        while not rospy.is_shutdown():
			self.send_vel_cmd()
		    	rate.sleep()

    def update_desired_pos(self, transformstamped_msg):
        """Callback function for sub_desired_pos topic, stores the 
        desired position and also calculates its velocity

        Args:
        tranformstamped_msg: msg received from subscriber

        """

        #store desired position
        self.desired_pos = transformstamped_msg


        pos_act = np.array([self.desired_pos.transform.translation.x,
                            self.desired_pos.transform.translation.y,
                            self.desired_pos.transform.translation.z,
                            self.desired_pos.transform.rotation.z])

        tdes_act = self.desired_pos.header.stamp.to_sec() + \
        self.desired_pos.header.stamp.to_nsec()/(10**9)

        pos_prev = np.array([self.desired_pos_prev.transform.translation.x,
                            self.desired_pos_prev.transform.translation.y,
                            self.desired_pos_prev.transform.translation.z,
                            self.desired_pos_prev.transform.rotation.z])
        
        tdes_prev = self.desired_pos_prev.header.stamp.to_sec() + \
        self.desired_pos_prev.header.stamp.to_nsec()/(10**9)


        try:
            delta = pos_act - pos_prev
            #Ensure difference in yaw doesn't exceed pi
            if delta[3] > np.pi:
                delta[3] = delta[3] - 2 * np.pi
            elif delta[3] < - np.pi:
                delta[3] = delta[3] + 2 * np.pi 
            #calculate velocity
            self.desired_vel = delta/float(tdes_act - tdes_prev)
        except:
            ropsy.logwarn('Division by zero encountered when calculating \
                desired velocity')
            pass

        self.desired_pos_prev = self.desired_pos


    def update_flat_in(self, msg):
		self.flat_input = list(msg.flat_input)

    def update_flat_state_op(self, msg):
		self.flat_state_op = list(msg.flat_state)

    def update_quadrotor_state(self, transfrom_stamped_msg):
		"""Callback function for sub_vicon topic, stores the quad rotor
		position and also calculates its velocity 

		Args:
			tranform_stamped_msg: msg received from subscriber
		"""

		#store received msg
		self.actual_pos = transfrom_stamped_msg

		#calculate velocity
		rotation = (self.actual_pos.transform.rotation.x,
		    self.actual_pos.transform.rotation.y,
		    self.actual_pos.transform.rotation.z,
		    self.actual_pos.transform.rotation.w)
		euler_angle = euler_from_quaternion(rotation)

		psi_act = euler_angle[2]

		pos_act = np.array([self.actual_pos.transform.translation.x,
						  self.actual_pos.transform.translation.y,
						  self.actual_pos.transform.translation.z,
						  psi_act])

		t_act = self.actual_pos.header.stamp.to_sec() + \
							self.actual_pos.header.stamp.to_nsec()/(10**9)

		rotation = (self.actual_pos_prev.transform.rotation.x,
		    self.actual_pos_prev.transform.rotation.y,
		    self.actual_pos_prev.transform.rotation.z,
		    self.actual_pos_prev.transform.rotation.w)
		euler_angle = euler_from_quaternion(rotation)

		psi_prev = euler_angle[2]

		pos_prev = np.array([self.actual_pos_prev.transform.translation.x,
							 self.actual_pos_prev.transform.translation.y,
							 self.actual_pos_prev.transform.translation.z,
							 psi_prev])

		t_prev = self.actual_pos_prev.header.stamp.to_sec() + \
						self.actual_pos_prev.header.stamp.to_nsec()/(10**9)
		

		try:
			delta = pos_act - pos_prev
			#Ensure difference in yaw doesn't exceed pi
			if delta[3] > np.pi:
				delta[3] = delta[3] - 2 * np.pi 
			elif delta[3] < - np.pi:
				delta[3] = delta[3] + 2 * np.pi 

			#calculate velocity
			self.actual_vel = delta/float(t_act - t_prev)
			
		except:
			ropsy.logwarn('Division by zero encountered when calculating \
									actual velocity')
			pass


		self.actual_pos_prev = self.actual_pos

		#update flat state:
		phi_act = euler_angle[0]
		theta_act = euler_angle[1]
		psi_act = euler_angle[2]
		
		x_dd = 9.81*(sin(psi_act)*tan(phi_act)/cos(theta_act) + cos(psi_act)*tan(theta_act))
		y_dd = 9.81*(-cos(psi_act)*tan(phi_act)/cos(theta_act) + sin(psi_act)*tan(theta_act))

		
		self.flat_state[0] = self.actual_pos.transform.translation.x
		self.flat_state[1] = self.actual_pos.transform.translation.y
		self.flat_state[2] = self.actual_pos.transform.translation.z
		self.flat_state[3] = (self.actual_vel[0] + self.actual_vel_prev[0] + self.actual_vel_prev2[0] + self.actual_vel_prev3[0])/4.0
		self.flat_state[4] = (self.actual_vel[1] + self.actual_vel_prev[1] + self.actual_vel_prev2[1] + self.actual_vel_prev3[1])/4.0
		self.flat_state[5] = (self.actual_vel[2] + self.actual_vel_prev[2] + self.actual_vel_prev2[2] + self.actual_vel_prev3[2])/4.0
		self.flat_state[6] = x_dd
		self.flat_state[7] = y_dd
		self.flat_state[8] = psi_act

		self.flat_state_msg.flat_state = self.flat_state

		#publish the flat_state
		self.pub_flat_state.publish(self.flat_state_msg)

		self.actual_vel_prev3 = self.actual_vel_prev2
		self.actual_vel_prev2 = self.actual_vel_prev
		self.actual_vel_prev = self.actual_vel


    def psi_inv(self, v, z):
		""" Function that computes the input u from flat state and flat
		input 

		Args:
		    z: flat state
		    v: flat input
		Returns:
		    u: input
		"""
		z_d = z[5]
		x_dd = z[6]
		y_dd = z[7]
		psi = z[8]

		x_ddd = v[0]
		y_ddd = v[1]
		z_dd = v[2] #set z to zeroo
		psi_d = v[3]

		g = 9.81

		#z controller
		## Ignore z
		z_d_c =  z_dd / 1.0 +z_d # set z to zeroo


		#calculate theta and phi
		lamda = 1/(z_dd + g) * (cos(psi)*x_dd + sin(psi)*y_dd)
		theta = np.arctan(lamda)
		alpha = cos(theta)/(z_dd + g) * (sin(psi)*x_dd - cos(psi)*y_dd)
		phi = np.arctan(alpha)


		#calculate commanded theta
		lamda_xdd = cos(psi)/(z_dd + g)
		lamda_ydd = sin(psi)/(z_dd + g)
		lamda_psi = 1/(z_dd + g)*(-sin(psi)*x_dd + cos(psi)*y_dd)
		lamda_d = lamda_xdd * x_ddd + lamda_ydd * y_ddd + \
				    lamda_psi * psi_d

		theta_d = 1/(1+lamda**2) * lamda_d

		theta_c  = 1.2*(1/0.5 * theta_d + theta)

		#calculate commanded phi
		alpha_theta = -sin(theta)/(z_dd + g)*(sin(psi)*x_dd  - cos(psi)* y_dd)
		alpha_xdd = cos(theta)/(z_dd + g)*sin(psi)
		alpha_ydd = -cos(theta)/(z_dd + g)*cos(psi)
		alpha_psi = cos(theta)/(z_dd + g)*(cos(psi)*x_dd + sin(psi)*y_dd)

		alpha_d = alpha_theta * theta_d + alpha_xdd * x_ddd + alpha_ydd * y_ddd \
				    + alpha_psi * psi_d
		phi_d = 1/(1+alpha**2) * alpha_d		

		phi_c = 1.2* (1/0.5 * phi_d + phi)

		return np.array([z_d_c, phi_c, theta_c, psi_d])


    def send_vel_cmd(self):
		"""Function that sends velocity commands to ardrone 
		"""
		#PID controller
		#calculate commands using position controller
		# phi_c, theta_c, z_d, psi_d = self.pos_controller.pos_cont(self.actual_pos, \
		# 	self.desired_pos, self.actual_vel, self.desired_vel)

    	#FMPC controller
		u = self.psi_inv(self.flat_input, self.flat_state_op)

		#Propagate flat_state_op forward in time
		self.flat_state_op [0] += self.flat_state_op [3] * self.dt
		self.flat_state_op [1] += self.flat_state_op [4] * self.dt
		self.flat_state_op [2] += self.flat_state_op [5] * self.dt
		self.flat_state_op [3] += self.flat_state_op [6] * self.dt
		self.flat_state_op [4] += self.flat_state_op [7] * self.dt
		self.flat_state_op [5] += self.flat_input [2] * self.dt
		self.flat_state_op [6] += self.flat_input [0] * self.dt
		self.flat_state_op [7] += self.flat_input [1] * self.dt
		self.flat_state_op [8] += self.flat_input [3] * self.dt	
		
		#Assign inputs
		phi_c = u[1]
		theta_c = u[2]
		z_d= u[0]
		psi_d = u[3]

		#set velocity commands
		vel_cmd_msg = Twist()
		vel_cmd_msg.linear.x  = phi_c
		vel_cmd_msg.linear.y  = theta_c
		vel_cmd_msg.linear.z  = z_d
		vel_cmd_msg.angular.z  = psi_d
		

		#publish velocity commands
		self.pub_vel_cmd.publish(vel_cmd_msg)		


if __name__ == '__main__':
    # write code to create ROSControllerNode
    rospy.init_node('ros_interface')
    ROSControllerNode()
    #rospy.spin()


