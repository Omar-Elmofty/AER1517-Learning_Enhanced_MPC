#!/usr/bin/env python2

"""Class for writing MPC controller"""

from __future__ import division, print_function, absolute_import

# Import ROS libraries
import numpy as np
import roslib
import rospy
import numpy as np
import math 
import time


# Import class that computes the desired positions
from cvxopt import matrix, solvers
from numpy import sin,cos,tan

from aer1217_ardrone_simulator.msg import FlatState
from aer1217_ardrone_simulator.msg import FlatInput
from std_msgs.msg import Int16



class MPC(object):
    def __init__(self):

        #publisher
        self.flat_in_pub = rospy.Publisher('/aer1217/flat_in', 
                                    FlatInput, queue_size=32)

        self.flat_state_op_pub = rospy.Publisher('/aer1217/flat_state_op', 
                                    FlatState, queue_size=32)

        #Subscribers
        self.flat_state_sub = rospy.Subscriber('/aer1217/flat_state', 
                                        FlatState,
                                        self.update_flat_state)

        self.des_path_sub = rospy.Subscriber('/aer1217/desired_path_dnn', 
                                      FlatState, self.update_des_path_dnn)

        self.desired_path_sub = rospy.Subscriber('/aer1217/desired_path', 
                                      FlatState, self.update_desired_unmod_path)

        self.comp_sub = rospy.Subscriber('/aer1217/send_complete', 
                                     Int16, self.start_follow)
	   
        #FLAG FOR RUNNING WITH DNN
        self.with_DNN = False

        #publishers for learning
        self.learning_state_pub = rospy.Publisher('/aer1217/learning_state', 
                                    FlatState, queue_size=32)
        self.learning_input_pub = rospy.Publisher('/aer1217/learning_input', 
                                    FlatState, queue_size=32)

        #Publishing rate
        self.rate = rospy.Rate(10)

        #Global variables
        self.flat_state = FlatState().flat_state
        self.flat_input_msg = FlatInput()
        self.flat_state_op_msg = FlatState()
        self.learning_state_msg = FlatState()
        self.learning_input_msg = FlatState()

        #save all msgs from this list
        self.des_path = []
        self.desired_path_unmod = []
        self.start_msg = 0
        self.rrt_tidx = 0
        self.start_flag = 0

        #Initialize MPC variables
        self.dt = 1/10.0 #delta used for MPC
        self.N = 20 #prediction horizon
        self.z_des = np.zeros((self.N, 9)) # initialize desired position
        self.z_des[:,2] = 2*np.ones(self.N) #initialize z positions to 2
        self.Q = np.diag([100, 100, 10, 0, 0, 0, 0, 0, 10])
        self.R = np.diag([1, 1, 1, 1])

        #Initialize MPC optimization matrices
        self.P, self.G, self.h, self.A = self.initalize_cvxopt_matrices()

        #silent solver 
        solvers.options['show_progress'] = False

    def update_des_path_dnn(self, msg):
        self.des_path.append(list(msg.flat_state))

    def update_desired_unmod_path(self, msg):
        self.desired_path_unmod.append(list(msg.flat_state))

    def update_flat_state(self, msg):
        self.flat_state = msg.flat_state

    def start_follow(self, msg):
        self.start_msg = msg

    def update_z_des(self):
        """Function that updates the z_desired vector used by MPC
        """

        self.z_des[0:self.N-1,:] = self.z_des[1:self.N,:]

        if self.start_msg > 0 : 
            if not(self.with_DNN):
                self.des_path = self.desired_path_unmod
            if (self.rrt_tidx + 2) == self.N:
                self.start_flag = 1
            self.z_des[self.N-1, :] = np.array(self.des_path[self.rrt_tidx])
            self.z_des[self.N-1, 2] = 2  #set z to 2
            self.rrt_tidx += 1
            self.rrt_tidx = self.rrt_tidx % len(self.des_path)
      

    def initalize_cvxopt_matrices(self):
        """Functions that creates the matrices required for solving the 
        quadratic program
        """

        #assign local variables for for simplicity
        N = self.N
        R = self.R
        Q = self.Q
        dt = self.dt

        #Define P matrix
        P = np.identity(N*(4+9))

        for i in range(N):
            P[i*4:i*4+4, i*4:i*4+4] = R
            P[N*4+i*9:N*4+i*9+9, N*4+i*9 : N*4+i*9+9] = Q


        #Define G & H matrices
        G = np.zeros((N*(4+9),N*(4+9)))
        h = np.zeros((N*(4+9),1))


        #Define A matrix
        A1 = np.zeros((9,4))
        A1[5,2] = -dt
        A1[6,0] = -dt
        A1[7,1] = -dt
        A1[8,3] = -dt

        A2 = - np.identity(9)
        A2[0,3] = -dt
        A2[1,4] = -dt
        A2[2,5] = -dt
        A2[3,6] = -dt
        A2[4,7] = -dt


        A = np.zeros((N*9,N*(4+9)))
        A[:,N*4:] = np.identity(N*9)


        for i in range(N):
            A[i*9:i*9+9, i*4:i*4+4] = A1
            if i != 0:
                A[i*9:i*9+9, N*4+(i-1)*9:N*4+(i-1)*9+9 ] = A2

        #Convert all arrays to matrix format
        P = matrix(P)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        
        return P, G, h, A


    def run_MPC(self, z0):
        """Function that runs that solves the MPC quadratic program
        Arg's:
            z0: Current Quad state
        Returns:
            vk: optimal flat input
            zk: optimal flat output
        """

        #set time step and prediction horizon
        dt = self.dt
        N = self.N

        #Define q vector
        q = np.zeros((N*(4+9),1))
        for i in range(N):
            q[N*4+i*9:N*4+i*9+9, 0] = -self.Q.dot(self.z_des[i,:].reshape(-1,1)).flatten()

        #define b_vector
        b = np.zeros((N*9,1))
    
        b[0,0] = z0[0] + dt * z0[3]
        b[1,0] = z0[1] + dt * z0[4]
        b[2,0] = z0[2] + dt * z0[5]
        b[3,0] = z0[3] + dt * z0[6]
        b[4,0] = z0[4] + dt * z0[7]
        b[5,0] = z0[5] 
        b[6,0] = z0[6]
        b[7,0] = z0[7] 
        b[8,0] = z0[8]  
        
        #convert to matrix format
        b = matrix(b)
        q = matrix(q)

        #solve quadratic program
        sol = solvers.qp(self.P, q, self.G, self.h, self.A, b) 
        
        #extract solutions
        vk = sol['x'][0:4]
        zk = sol['x'][N*4:N*4+9]

        return vk, zk


    def spin_MPC(self):
        """Function that spins the MPC loop for controlling the quad copter 
        """

        #Initialize input
        vk = np.zeros(4)
        episode = 0
        idx = 0
        error_des = []
        error_act = []

        max_episodes = 10

        while not rospy.is_shutdown():
          
            if self.start_flag and episode <max_episodes:

                #Publish states and inputs for training the DNN
                self.learning_state_msg.flat_state = self.flat_state
                self.learning_input_msg.flat_state = self.des_path[idx]
                self.learning_state_pub.publish(self.learning_state_msg)
                self.learning_input_pub.publish(self.learning_input_msg)

                #save error for reporting
                error_des.append((self.flat_state[0]-self.desired_path_unmod[idx][0])**2 + \
                        (self.flat_state[1]-self.desired_path_unmod[idx][1])**2 )

                error_act.append( (self.flat_state[0]-self.des_path[idx][0])**2 + \
                        (self.flat_state[1]-self.des_path[idx][1])**2 )

                idx += 1

                #print error
                if idx % len(self.des_path) == 0:
                    episode +=1
                    idx =0
                    error_des = np.array(error_des)
                    error_act = np.array(error_act)
                    print('Episode num = ', episode)
                    print('Total RMS des= ', np.sqrt(np.mean(error_des)))
                    print('Total RMS act= ', np.sqrt(np.mean(error_act)))
                    error_des = []
                    error_act = []
                if episode == max_episodes:
                    print('episodes done')

            #Propagate the state one time step forward
            f_state = np.zeros(9)
            f_state[0] = self.flat_state[0] + self.flat_state[3]*self.dt
            f_state[1] = self.flat_state[1] + self.flat_state[4]*self.dt
            f_state[2] = self.flat_state[2] + self.flat_state[5]*self.dt
            f_state[3] = self.flat_state[3] + self.flat_state[6]*self.dt
            f_state[4] = self.flat_state[4] + self.flat_state[7]*self.dt
            f_state[5] = self.flat_state[5] + vk[2]*self.dt
            f_state[6] = self.flat_state[6] + vk[0]*self.dt
            f_state[7] = self.flat_state[7] + vk[1]*self.dt
            f_state[8] = self.flat_state[8] + vk[3]*self.dt

            #update desired state
            self.update_z_des()

            #Solve quadratic program
            vk, zk = self.run_MPC(f_state)

            #publish vk and zk
            self.flat_input_msg.flat_input = list(vk)
            self.flat_state_op_msg.flat_state = list(zk)
            self.flat_in_pub.publish(self.flat_input_msg)
            self.flat_state_op_pub.publish(self.flat_state_op_msg)
            
            #force sleep rate
            self.rate.sleep()



if __name__ == '__main__':
    # write code to create ROSControllerNode
    rospy.init_node('MPC')
    mpc = MPC()
    mpc.spin_MPC()