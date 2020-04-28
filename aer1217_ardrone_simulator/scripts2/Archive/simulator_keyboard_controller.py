#!/usr/bin/env python

# The Keyboard Controller Node for the tutorial "Up and flying with the AR.Drone and ROS | Getting Started"
# https://github.com/mikehamer/ardrone_tutorials

# This controller extends the base DroneVideoDisplay class, adding a keypress handler to enable keyboard control of the drone

# Import the ROS libraries, and load the manifest file which through <depend package=... /> will give us access to the project dependencies
import roslib
import rospy

# Load the DroneController class, which handles interactions with the drone, and the DroneVideoDisplay class, which handles video display
from drone_controller import BasicDroneController
from simulator_video_display import DroneVideoDisplay
from ardrone_autonomy.srv import *
from aer1217_ardrone_simulator.srv import *
import numpy as np
from std_msgs.msg import Int16

# Finally the GUI libraries
from PySide import QtCore, QtGui


# Here we define the keyboard map for our controller (note that python has no enums, so we use a class)
class KeyMapping(object):
	PitchForward     = QtCore.Qt.Key.Key_E
	PitchBackward    = QtCore.Qt.Key.Key_D
	RollLeft         = QtCore.Qt.Key.Key_S
	RollRight        = QtCore.Qt.Key.Key_F
	YawLeft          = QtCore.Qt.Key.Key_W
	YawRight         = QtCore.Qt.Key.Key_R
	IncreaseAltitude = QtCore.Qt.Key.Key_Q
	DecreaseAltitude = QtCore.Qt.Key.Key_A
	Takeoff          = QtCore.Qt.Key.Key_Y
	Land             = QtCore.Qt.Key.Key_H
	# 2017-03-22 switch to bottom camera through rosservice, specify camera channel as 1
	CamChange        = QtCore.Qt.Key.Key_X
	ProcessImg       = QtCore.Qt.Key.Key_P
	###
	Emergency        = QtCore.Qt.Key.Key_Space
	ThrottleCut      = QtCore.Qt.Key.Key_K # kill switch

	Lin_move      = QtCore.Qt.Key.Key_N
	Cir_move      = QtCore.Qt.Key.Key_M


# Our controller definition, note that we extend the DroneVideoDisplay class
class KeyboardController(DroneVideoDisplay):
	def __init__(self):
		super(KeyboardController,self).__init__()
		
		self.pitch = 0
		self.roll = 0
		self.yaw_velocity = 0 
		self.z_velocity = 0
		self.camchannel = 0
		self.processImagesBool = False # disable image processing at start-up
		

		self.switch_moves = 0 #0 is for linear move, and 1 for circular move

		self.pub_move = rospy.Publisher('/aer1217/move_type',
						Int16,queue_size=32)

# We add a keyboard handler to the DroneVideoDisplay to react to keypresses
	def keyPressEvent(self, event):
		key = event.key()

		# If we have constructed the drone controller and the key is not generated from an auto-repeating key
		if controller is not None and not event.isAutoRepeat():
			# Handle the important cases first!
			if key == KeyMapping.ThrottleCut:
				controller.SendEmergency()
			elif key == KeyMapping.Emergency:
				controller.SendLand() # modify to prevent damage to quad
			elif key == KeyMapping.Takeoff:
				controller.SendTakeoff()		
			elif key == KeyMapping.Land:
				controller.SendLand()	
			elif key == KeyMapping.CamChange:
				rospy.wait_for_service('ardrone/togglecam')
				try:
				    switchcam = rospy.ServiceProxy('ardrone/togglecam', ToggleCam)
				    switchcam()
				except rospy.ServiceException, e:
				    print "Service call failed: %s" % e
				    
			elif key == KeyMapping.ProcessImg:
				if self.processImagesBool == False:
					self.processImagesBool = True
					print "Image processing is turned on"
					super(KeyboardController,self).EnableImageProcessing()
				else:
					self.processImagesBool = False
					print "Image processing is turned off"
					super(KeyboardController,self).DisableImageProcessing()
			else:
				# Now we handle moving, notice that this section is the opposite (+=) of the keyrelease section
				if key == KeyMapping.YawLeft:
					self.yaw_velocity += 1
				elif key == KeyMapping.YawRight:
					self.yaw_velocity += -1

				elif key == KeyMapping.PitchForward:
					self.pitch += 1
				elif key == KeyMapping.PitchBackward:
					self.pitch += -1

				elif key == KeyMapping.RollLeft:
					self.roll += 1
				elif key == KeyMapping.RollRight:
					self.roll += -1

				elif key == KeyMapping.IncreaseAltitude:
					self.z_velocity += 1
				elif key == KeyMapping.DecreaseAltitude:
					self.z_velocity += -1
				elif key == KeyMapping.Lin_move:
					self.switch_moves = 0
					self.pub_move.publish(self.switch_moves)
				elif key == KeyMapping.Cir_move:
					self.switch_moves = 1
					self.pub_move.publish(self.switch_moves)

			# finally we set the command to be sent. The controller handles sending this at regular intervals
			controller.SetCommand(self.roll, self.pitch, self.yaw_velocity, self.z_velocity)


	def keyReleaseEvent(self,event):
		key = event.key()

		# If we have constructed the drone controller and the key is not generated from an auto-repeating key
		if controller is not None and not event.isAutoRepeat():
			# Note that we don't handle the release of emergency/takeoff/landing keys here, there is no need.
			# Now we handle moving, notice that this section is the opposite (-=) of the keypress section
			if key == KeyMapping.YawLeft:
				self.yaw_velocity -= 1
			elif key == KeyMapping.YawRight:
				self.yaw_velocity -= -1

			elif key == KeyMapping.PitchForward:
				self.pitch -= 1
			elif key == KeyMapping.PitchBackward:
				self.pitch -= -1

			elif key == KeyMapping.RollLeft:
				self.roll -= 1
			elif key == KeyMapping.RollRight:

			elif key == KeyMapping.IncreaseAltitude:
				self.z_velocity -= 1
			elif key == KeyMapping.DecreaseAltitude:
				self.z_velocity -= -1

			# finally we set the command to be sent. The controller handles sending this at regular intervals
			controller.SetCommand(self.roll, self.pitch, self.yaw_velocity, self.z_velocity)



# Setup the application
if __name__=='__main__':
	import sys
	# Firstly we setup a ros node, so that we can communicate with the other packages
	rospy.init_node('ardrone_keyboard_controller')

	# Now we construct our Qt Application and associated controllers and windows
	app = QtGui.QApplication(sys.argv)
	controller = BasicDroneController()
	display = KeyboardController()

	display.show()

	# executes the QT application
	status = app.exec_()

	# and only progresses to here once the application has been shutdown
	rospy.signal_shutdown('Great Flying!')
	sys.exit(status)
