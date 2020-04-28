#!/usr/bin/env python3

"""Class for writing MPC controller"""

from __future__ import division, print_function, absolute_import

# Import ROS libraries
import numpy as np
import roslib
import rospy
import numpy as np
import keras
import os, rospkg
from aer1217_ardrone_simulator.msg import FlatState
from std_msgs.msg import Int16
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from min_snap import min_snap_trajectory
from geometry_msgs.msg import TransformStamped


class Inference(object):
    def __init__(self):

        #publisher
        self.pub_des_path = rospy.Publisher('/aer1217/desired_path_dnn', 
                                      FlatState, queue_size=32)

        self.pub_send_comp = rospy.Publisher('/aer1217/send_complete', 
                                    Int16,
                                    queue_size=32)

        self.pub_desired_path_unmod = rospy.Publisher('/aer1217/desired_path', 
                                      FlatState, queue_size=32)


        #For PID
        # Publishers
        self.pub_des_pos = rospy.Publisher('/aer1217/desired_position', 
                                      TransformStamped, queue_size=32)

        #Initialize msgs
        self.desired_position_msg = TransformStamped()
        self.desired_path_msg = FlatState()
        self.desired_unmod_msg = FlatState()


        #Normalization coefficients
        self.max_pos=  1.1148116815609779
        self.min_pos=  -1.0464868543970438
        self.max_vel=  6.608598081092131
        self.min_vel=  -5.94381835900806
        self.max_acc=  6.861119716920872
        self.min_acc=  -5.886668443124795

        print('**make sure that the min max normalization coefficients are correct**')

        #Keras model
        self.model = None


    def compile_inference_data(self, X):
        """Functions that compiles all the inference data that will be 
        input to the DNN

        Args:
            X: Minimum snap trajectory
        Returns:
            x_infer: array containing all inference data
        """

        x_infer = []
        #extract data from X
        for i in range(X.shape[1]-1):
            x = [0,
                0,
                X[3,i],
                X[4,i],
                X[6,i],
                X[7,i],
                X[0,i+1] - X[0,i],
                X[1,i+1] - X[1,i],
                X[3,i+1],
                X[4,i+1],
                X[6,i+1],
                X[7,i+1]]

            x_infer.append(x)

        #convert to numpy array
        x_infer = np.array(x_infer)

        #normalize
        x_infer[:,6:8] = (x_infer[:,6:8] - self.min_pos) / (self.max_pos - self.min_pos)

        x_infer[:,2:4] = (x_infer[:,2:4] - self.min_vel) / (self.max_vel - self.min_vel)
        x_infer[:,8:10] = (x_infer[:,8:10] - self.min_vel) / (self.max_vel - self.min_vel)

        x_infer[:,4:6] = (x_infer[:,4:6] - self.min_acc) / (self.max_acc - self.min_acc)
        x_infer[:,10:12] = (x_infer[:,10:12] - self.min_acc) / (self.max_acc - self.min_acc)

        return x_infer


    def inference(self, X):
        """Function that perform inference using the DNN
        Args:
            X: minimum snap trajectory
        Returns:
            x: list of x positions (output from DNN)
            y: list of y positions (output from DNN)
            x_des: x values of the unmodified trajectory (before DNN)
            y_des: y values of the unmodified trajectory (before DNN)

        """

        #use model to predict
        x_infer = self.compile_inference_data(X)

        #plot inference data
        predictions = self.model.predict(x_infer)

        #convert predictions to actual positions
        predictions = predictions * (self.max_pos - self.min_pos) + self.min_pos

        #create trajectory
        x = [0]
        y = [0]


        for i in range(2, len(predictions)):
            x_avg = 1/3. *(predictions[i][0] + X[0,i] \
                + predictions[i-1][2] + X[0,i-1]  \
                + predictions[i-2][4] + X[0,i-2])

            y_avg = 1/3. *(predictions[i][1] + X[1,i] \
                + predictions[i-1][3] + X[1,i-1]  \
                + predictions[i-2][5] + X[1,i-2])

            x.append(x_avg)
            y.append(y_avg)

        x_des = []
        y_des = []
        #compile actual trajectory
        for i in range(X.shape[1]):
            x_des.append(X[0,i])
            y_des.append(X[1,i])


        
        return x, y, x_des, y_des

    def plot_results(self, x, y, x_des, y_des):

        #plot trajectory
        plt.plot(x,y, label='inference result')
        plt.plot(x_des, y_des, label='desired traj')
        plt.legend()
        plt.title('Path that will be published - Close window to start publishing')
        plt.show()

    def load_saved_model(self):
        """Function that loads the saved model
        """
        rospack = rospkg.RosPack()
        dirc = os.path.join(rospack.get_path("aer1217_ardrone_simulator"), "DNN/model.json")
        # load json and create model
        json_file = open(dirc, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        dirc = os.path.join(rospack.get_path("aer1217_ardrone_simulator"), "DNN/model.h5")
        self.model.load_weights(dirc)
        print("Loaded model from disk")
         
        # evaluate loaded model on test data
        self.model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['acc', 'mae'])

    def publish_path(self, x, y, x_des, y_des):
        """Function that publishes path to MPC
        Args:
            x: list of x positions (output from DNN)
            y: list of y positions (output from DNN)
            x_des: x values of the unmodified trajectory (before DNN)
            y_des: y values of the unmodified trajectory (before DNN)
        """

        rate = rospy.Rate(10)

        for i in range(len(x)):

            #For MPC
            state = np.zeros(9)
            state[0] = x[i]
            state[1] = y[i]
            self.desired_path_msg.flat_state = list(state)
            self.pub_des_path.publish(self.desired_path_msg)
            desired_point = np.zeros(9)
            desired_point[0] = x_des[i]
            desired_point[1] = y_des[i]
            self.desired_unmod_msg.flat_state = list(desired_point)
            self.pub_desired_path_unmod.publish(self.desired_unmod_msg)

            #For PID
            self.desired_position_msg.transform.translation.x = x_des[i]
            self.desired_position_msg.transform.translation.y = y_des[i]
            self.desired_position_msg.transform.translation.z = 2
            self.desired_position_msg.transform.rotation.z = 0
            self.desired_position_msg.header.stamp = rospy.Time.now()
            self.pub_des_pos.publish(self.desired_position_msg)

            rate.sleep()

        print('Path_published')

        #Publish flag indicating full path was sent
        comp_send = 1
        self.pub_send_comp.publish(comp_send)



if __name__ == '__main__':

    # write code to create ROSControllerNode
    rospy.init_node('train_model')
    infer = Inference()     

    infer.load_saved_model()

    state_init = np.array([0,0,2,0,0,0,0,0,0])
    state_final = np.array([0,0,2,0,0,0,0,0,0])

    #Trajectory 1
    #int_points = [np.array([3,-3,2]),np.array([6,0,2]), np.array([3,3,2])]

    #Trajectory 2
    #int_points = [np.array([6,3,2]),np.array([2,0,2]), np.array([6,-3,2])]

    #Trajectory 3
    int_points = [np.array([-3,-3,2]),np.array([0,-1,2]), np.array([3,-3,2]), 
    np.array([1,0,2]),np.array([3,3,2]),np.array([0,1,2]),np.array([-3,3,2]) ]    

    s, X = min_snap_trajectory(state_init, state_final, int_points, 1/10.0, speed=2.0)
    print(X.shape[1])
    x, y, x_des, y_des = infer.inference(X)

    infer.plot_results(x, y, x_des, y_des)

    infer.publish_path(x, y, x_des, y_des)
    rospy.spin()