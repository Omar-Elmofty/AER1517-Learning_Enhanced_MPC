#!/usr/bin/env python3

"""Class for writing MPC controller"""

from __future__ import division, print_function, absolute_import

# Import ROS libraries
import numpy as np
import roslib
import rospy
from aer1217_ardrone_simulator.msg import FlatState
import matplotlib.pyplot as plt

class TrainData(object):
    def __init__(self):

        #Subscribers
        self.learning_state_sub = rospy.Subscriber('/aer1217/learning_state', 
                                        FlatState,
                                        self.update_learning_state)
        self.learning_input_sub = rospy.Subscriber('/aer1217/learning_input', 
                                        FlatState,
                                        self.update_learning_input)

        #Global variables
        self.learned_path_msg = FlatState()
        self.learning_state_list = []
        self.learning_input_list = []
        self.last_msg_time = rospy.Time.now().to_sec()

    def update_learning_state(self, msg):
        self.learning_state_list.append(list(msg.flat_state))
        self.last_msg_time = rospy.Time.now().to_sec()

    def update_learning_input(self, msg):
        self.learning_input_list.append(list(msg.flat_state))

    def compile_training_data(self):
        """Function that compiles training data
        Returns: 
            x_train: array containing all training cases
            y_train: array containing all labels
        """

        x_train = []
        y_train = []

        #check that the lengths of states and inputs is the same

        if len(self.learning_state_list) != len(self.learning_input_list):
            min_length = min(len(self.learning_state_list), len(self.learning_input_list))
            self.learning_state_list = self.learning_state_list[0:min_length]
            self.learning_input_list = self.learning_input_list[0:min_length]
            print('WARNING: Truncated states and inputs')

        for i in range(len(self.learning_state_list)-3):

            x = [0,
                0,
                self.learning_state_list[i][3],
                self.learning_state_list[i][4],
                self.learning_state_list[i][6],
                self.learning_state_list[i][7],
                self.learning_state_list[i+1][0] - self.learning_state_list[i][0],
                self.learning_state_list[i+1][1] - self.learning_state_list[i][1],
                self.learning_state_list[i+1][3],
                self.learning_state_list[i+1][4],
                self.learning_state_list[i+1][6],
                self.learning_state_list[i+1][7]]

            y = [self.learning_input_list[i+1][0]- self.learning_state_list[i][0],
                self.learning_input_list[i+1][1] - self.learning_state_list[i][1],
                self.learning_input_list[i+2][0]- self.learning_state_list[i][0],
                self.learning_input_list[i+2][1] - self.learning_state_list[i][1],
                self.learning_input_list[i+3][0]- self.learning_state_list[i][0],
                self.learning_input_list[i+3][1] - self.learning_state_list[i][1]
                ]

            x_train.append(x)
            y_train.append(y)

        #convert x_train & y_train to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)


        return x_train, y_train

        
        

if __name__ == '__main__':
    # write code to create ROSControllerNode
    rospy.init_node('train_data')
    train = TrainData()

    print('waiting for first msg')
    rospy.wait_for_message("/aer1217/learning_state", FlatState)
    print('Loading data')

    current_time = rospy.Time.now().to_sec() 
    while (current_time - train.last_msg_time) < 5 :
        current_time =  rospy.Time.now().to_sec()
   
    x_train, y_train = train.compile_training_data()

    np.save('x_train.npy', x_train)
    np.save('y_train.npy', y_train)
    print('Data saved')
     