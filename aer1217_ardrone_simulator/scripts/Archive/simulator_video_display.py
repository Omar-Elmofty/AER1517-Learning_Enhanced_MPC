#!/usr/bin/env python

# A basic video display window for the tutorial "Up and flying with the AR.Drone and ROS | Getting Started"
# https://github.com/mikehamer/ardrone_tutorials_getting_started

# This display window listens to the drone's video feeds and updates the display at regular intervals
# It also tracks the drone's status and any connection problems, displaying them in the window's status bar
# By default it includes no control functionality. The class can be extended to implement key or mouse listeners if required

# Import the ROS libraries, and load the manifest file which through <depend package=... /> will give us access to the project dependencies
import roslib
import rospy
import cv2
import math

# Import the two types of messages we're interested in
from sensor_msgs.msg import Image    	 # for receiving the video feed

# We need to use resource locking to handle synchronization between GUI thread and ROS topic callbacks
from threading import Lock

# The GUI libraries
from PySide import QtCore, QtGui
from aer1217_ardrone_simulator.srv import *

# 2017-03-22 Import libraries from OpenCV
# OpenCV Bridge http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from tf.transformations import euler_from_quaternion
import numpy as np


# Some Constants
CONNECTION_CHECK_PERIOD = 250 #ms
GUI_UPDATE_PERIOD = 20 #ms
DETECT_RADIUS = 4 # the radius of the circle drawn when a tag is detected


class DroneVideoDisplay(QtGui.QMainWindow):
	
	def __init__(self):
		# Construct the parent class
		super(DroneVideoDisplay, self).__init__()

		# Setup our very basic GUI - a label which fills the whole window and holds our image
		self.setWindowTitle('Video Feed')
		self.imageBox = QtGui.QLabel(self)
		self.setCentralWidget(self.imageBox) 
		
		# Subscribe to the drone's video feed, calling self.ReceiveImage when a new frame is received
		self.subVideoBottom   = rospy.Subscriber('/ardrone/bottom/image_raw/',Image,self.ReceiveImageBottom,queue_size=1)
		self.subVideoFront   = rospy.Subscriber('/ardrone/front/image_raw/',Image,self.ReceiveImageFront,queue_size=1)

		self.camera = 0 # start with front first
		
		# Holds the image frame received from the drone and later processed by the GUI
		self.image = None
		self.imageLock = Lock()

		self.tags = []
		self.tagLock = Lock()
				
		# Holds the status message to be displayed on the next GUI update
		self.statusMessage = ''

		# Tracks whether we have received data since the last connection check
		# This works because data comes in at 50Hz but we're checking for a connection at 4Hz
		self.communicationSinceTimer = False
		self.connected = False

		# A timer to check whether we're still connected
		self.connectionTimer = QtCore.QTimer(self)
		self.connectionTimer.timeout.connect(self.ConnectionCallback)
		self.connectionTimer.start(CONNECTION_CHECK_PERIOD)
		
		# A timer to redraw the GUI
		self.redrawTimer = QtCore.QTimer(self)
		self.redrawTimer.timeout.connect(self.RedrawCallback)
		self.redrawTimer.start(GUI_UPDATE_PERIOD)
		
		# 2017-03-22 convert ROS images to OpenCV images
		self.bridge = CvBridge()

		# 2017-03-31 Lab 4 processing images variables
		self.processImages = False		
		self.cv_output = None
		self.cv_img = None
		
		rospy.Service('/ardrone/togglecam',ToggleCam,self.ToggleFrontBottomCamera)

	# Called every CONNECTION_CHECK_PERIOD ms, if we haven't received anything since the last callback, will assume we are having network troubles and display a message in the status bar
	def ConnectionCallback(self):
		self.connected = self.communicationSinceTimer
		self.communicationSinceTimer = False

	def RedrawCallback(self):
		if self.image is not None:
			# We have some issues with locking between the display thread and the ros messaging thread due to the size of the image, so we need to lock the resources
			self.imageLock.acquire()
			try:			
					# Convert the ROS image into a QImage which we can display
					if self.processImages == False:
						image = QtGui.QPixmap.fromImage(QtGui.QImage(self.image.data, self.image.width, self.image.height, QtGui.QImage.Format_RGB888))						
					# display processed image when processing is enabled
					else:
						if self.cv_output is not None:					
							# convert from openCV output cv_output image back to ROS image (Optional for visualization purposes)
							img_msg = self.bridge.cv2_to_imgmsg(self.cv_output, encoding="bgr8")
							# convert to QImage to be displayed
							image = QtGui.QPixmap.fromImage(QtGui.QImage(img_msg.data, img_msg.width, img_msg.height, QtGui.QImage.Format_RGB888))
						else:
							image = QtGui.QPixmap.fromImage(QtGui.QImage(self.image.data, self.image.width, self.image.height, QtGui.QImage.Format_RGB888))		
					
			finally:
				self.imageLock.release()

			# We could  do more processing (eg OpenCV) here if we wanted to, but for now lets just display the window.
			image = image.scaledToWidth(480)
			self.resize(image.width(),image.height())
			self.imageBox.setPixmap(image)

		# Update the status bar to show the current drone status & battery level
		if self.image is None:
			self.statusBar().showMessage("Simulator not started")
		else:
			if self.camera == 0:
				self.statusBar().showMessage("Displaying front camera")
			if self.camera == 1:
				self.statusBar().showMessage("Displaying bottom camera")
		
		

	def ReceiveImageBottom(self,data):
		if self.camera == 1:
			# Indicate that new data has been received (thus we are connected)
			self.communicationSinceTimer = True

			# We have some issues with locking between the GUI update thread and the ROS messaging thread due to the size of the image, so we need to lock the resources
			self.imageLock.acquire()
			try:
				self.image = data # Save the ros image for processing by the display thread
				# 2017-03-22 we do not recommend saving images in this function as it might cause huge latency
			finally:
				self.imageLock.release()

	def ReceiveImageFront(self,data):
		if self.camera == 0:
			# Indicate that new data has been received (thus we are connected)
			self.communicationSinceTimer = True

			# We have some issues with locking between the GUI update thread and the ROS messaging thread due to the size of the image, so we need to lock the resources
			self.imageLock.acquire()
			try:
				self.image = data # Save the ros image for processing by the display thread
				# 2017-03-22 we do not recommend saving images in this function as it might cause huge latency
			finally:
				self.imageLock.release()

	# 2017-03-22 sample function of saving images.
	# TODO feel free to modify, you could use a timer to capture the image at a certain rate, or modify the keyboard_controller.py to capture the image through a key
	def SaveImage(self,data):
		# ensure not in the process of acquiring image
		if self.image is not None:
			if self.imageLock.acquire():
				try:
					# convert from ROS image to OpenCV image
					cv_image = self.bridge.imgmsg_to_cv2(self.image, desired_encoding="bgr8")
				except CvBridgeError as e:
					print "Image conversion failed: %s" % e
				# TODO use cv2.imwrite function to write images to local disk
				# might want to consider storing the corresponding vehicle attitude/position to geolocate target
				self.imageLock.release()
				        
	# 2017-03-31 Lab 4 code ================================
	# the codes written here serve as a guideline, it is not required that you use the code. Feel free to modify.

	def EnableImageProcessing(self):  # called from KeyboardController Key_P
		self.processImages = True

	def DisableImageProcessing(self): # called from KeyboardController Key_P
		self.processImages = False

	def ToggleFrontBottomCamera(self,req):
		if (self.camera == 1):
			self.camera = 0
			return "Displaying front camera"
		else:
			self.camera = 1
			return "Displaying bottom camera"
	

if __name__=='__main__':
	import sys
	rospy.init_node('sim_video_display')
	app = QtGui.QApplication(sys.argv)
	display = DroneVideoDisplay()
	display.show()
	status = app.exec_()
	rospy.signal_shutdown('Great Flying!')
	sys.exit(status)
