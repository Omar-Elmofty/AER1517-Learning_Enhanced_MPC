# Learning Enhanced Model Predictive Controller for Quadcopters

AER1517 - Controls for robotics - Final project 

Link for [project report](https://drive.google.com/open?id=12NddcTXf4h5ht1D1IKoDDKIvNOBbi42d)

Link for [Demonstration video](https://youtu.be/Rri49FRkjCo)

## Requirements 
[CVXOPT](https://cvxopt.org/)

[Keras](https://keras.io/)

[ROS Kinetic](http://wiki.ros.org/kinetic)

Preferable environment: Ubuntu 16.04

## Simulator Environment Setup
To setup the simulator environment, please follow the simulator setup instructions [here](https://drive.google.com/open?id=12NddcTXf4h5ht1D1IKoDDKIvNOBbi42d)

After the simulator is setup, perform the following:

`$ cd ~/aer1217/labs/src`

Remove the existing `aer1217_ardrone_simulator` package, and replace it with the package in this repository

`$ cd ~/aer1217/labs`

`$ catkin_make`

Consult [ROS Wiki](http://wiki.ros.org/Documentation) if you encounter any issues setting up the package

## Running FMPC

To run FMPC, use the following command:

`$ roslaunch aer1217_ardrone_simulator ardrone_simulator.launch` 

To run using the DNN output, change the flag (self.with_DNN) in `MPC.py` line 51 to True 

(`MPC.py` is located under `~/aer1217/labs/src/aer1217_ardrone_simulator/scripts`)


## Training the DNN

To record training data while running FMPC, run this command in separate terminal

`$ rosbag record /aer1217/learning_state /aer1217/learning_input`

To compile the training data:

`$ cd ~/aer1217/labs/src/aer1217_ardrone_simulator/DNN`

`$ rosrun aer1217_ardrone_simulator compile_training_data.py`

In a separate terminal

`$ rosbag play <bag file recorded>`

To train the DNN: 

`$ cd ~/aer1217/labs/src/aer1217_ardrone_simulator/DNN`

`$ python dnn_train.py`


## Acknowledgments 

Course material from UTIAS AER1517 - Controls for Robotics
    
Course material and drone simulator from UTIAS AER1217 - Autonomous UAS course 

Full references list can be found in the [final report](https://drive.google.com/open?id=12NddcTXf4h5ht1D1IKoDDKIvNOBbi42d)

Finally, special thanks for the authors of the following papers as it was influential for the work done in the project

Melissa Greeff and Angela P. Schoellig, Flatness-based Model
Predictive Control for Quadrotor Trajectory Tracking, Proc. of the
IEEE/RSJ International Conference on Intelligent Robots and Systems
(IROS), 2018, pp. 6740—6745

SiQi Zhou and Mohamed K. Helwa and Angela P. Schoellig, “Design
of Deep Neural Networks as Add-on Blocks for Improving Impromptu
Trajectory Tracking”, in Proc. of the IEEE Conference on Decision
and Control (CDC), 2017, pp 5201—5207


