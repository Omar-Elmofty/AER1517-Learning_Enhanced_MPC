#!/usr/bin/env python3

"""Class for writing MPC controller"""

from __future__ import division, print_function, absolute_import

# Import ROS libraries
import numpy as np
import roslib
import rospy
import numpy as np
import keras
from aer1217_ardrone_simulator.msg import FlatState
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

class Learner(object):
    def __init__(self):

        #publisher
        self.learned_path_pub = rospy.Publisher('/aer1217/learned_path', 
                                    FlatState, queue_size=32)

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
        #normalization variables
        self.max_pos = 0
        self.min_pos = 0
        self.max_vel = 0
        self.min_vel = 0
        self.max_acc = 0
        self.min_acc = 0
        #Keras model
        self.model = None


    def update_learning_state(self, msg):
        self.learning_state_list.append(list(msg.flat_state))
        self.last_msg_time = rospy.Time.now().to_sec()

    def update_learning_input(self, msg):
        self.learning_input_list.append(list(msg.flat_state))

    def compile_training_data(self):

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

    def normalize_data(self, x_train, y_train):
        #normalize the position data, velocities and accelerations
        self.max_pos = np.max(x_train[:,6:8])
        self.min_pos = np.min(x_train[:,6:8])
        self.max_vel = np.max(x_train[:,2:4])
        self.min_vel = np.min(x_train[:,2:4])
        self.max_acc = np.max(x_train[:,4:6])
        self.min_acc = np.min(x_train[:,4:6])

        x_train[:,6:8] = (x_train[:,6:8] - self.min_pos) / (self.max_pos - self.min_pos)
        y_train = (y_train - self.min_pos) / (self.max_pos - self.min_pos)

        x_train[:,2:4] = (x_train[:,2:4] - self.min_vel) / (self.max_vel - self.min_vel)
        x_train[:,8:10] = (x_train[:,8:10] - self.min_vel) / (self.max_vel - self.min_vel)

        x_train[:,4:6] = (x_train[:,4:6] - self.min_acc) / (self.max_acc - self.min_acc)
        x_train[:,10:12] = (x_train[:,10:12] - self.min_acc) / (self.max_acc - self.min_acc)

        return x_train, y_train


    def compile_inference_data(self):

        x_infer = []

        for i in range(len(self.learning_input_list)-1):
            x = [0,
                0,
                self.learning_input_list[i][3],
                self.learning_input_list[i][4],
                self.learning_input_list[i][6],
                self.learning_input_list[i][7],
                self.learning_input_list[i+1][0] - self.learning_input_list[i][0],
                self.learning_input_list[i+1][1] - self.learning_input_list[i][1],
                self.learning_input_list[i+1][3],
                self.learning_input_list[i+1][4],
                self.learning_input_list[i+1][6],
                self.learning_input_list[i+1][7]]

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


    def create_model(self):
        #model architecture
        model = Sequential()
        model.add(Dense(12, input_shape = (12,), activation = 'relu'))
        model.add(Dense(10, activation = 'relu'))
        model.add(Dense(6))
        #compile model
        sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['acc', 'mae'])

        model.summary()

        return model

    def train_model(self, x_train, y_train):

        self.model = self.create_model()
        print('x_train shape', x_train.shape)
        print('y_train shape', y_train.shape)
        print('max_x_train = ', np.max(x_train))

        history = self.model.fit(x_train,y_train,validation_split=0.1, batch_size = 10, epochs = 60)

        #save model
        # serialize model to JSON
        model_json = self.model.to_json()

        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")

        return history

    def inference(self):

        #use model to predict
        x_infer = self.compile_inference_data()
        x_train, y_train = self.compile_training_data()
        
        #plot training data
        predictions = self.model.predict(x_train)

        #convert predictions to actual positions
        predictions = predictions * (self.max_pos - self.min_pos) + self.min_pos

        #create trajectory
        x = [0]
        y = [0]



        for i in range(2, len(predictions)):
            x_avg = 1/3. *(predictions[i][0] + self.learning_state_list[i][0] \
                + predictions[i-1][2] + self.learning_state_list[i-1][0]  \
                + predictions[i-2][4] + self.learning_state_list[i-2][0])

            y_avg = 1/3. *(predictions[i][1] + self.learning_state_list[i][1] \
                + predictions[i-1][3] + self.learning_state_list[i-1][1]  \
                + predictions[i-2][5] + self.learning_state_list[i-2][1])

            x.append(x_avg)
            y.append(y_avg)

        plt.plot(x,y, label='training result')

        #plot inference data
        predictions = self.model.predict(x_infer)

        #convert predictions to actual positions
        predictions = predictions * (self.max_pos - self.min_pos) + self.min_pos

        #create trajectory
        x = [0]
        y = [0]


        for i in range(2, len(predictions)):
            x_avg = 1/3. *(predictions[i][0] + self.learning_input_list[i][0] \
                + predictions[i-1][2] + self.learning_input_list[i-1][0]  \
                + predictions[i-2][4] + self.learning_input_list[i-2][0])

            y_avg = 1/3. *(predictions[i][1] + self.learning_input_list[i][1] \
                + predictions[i-1][3] + self.learning_input_list[i-1][1]  \
                + predictions[i-2][5] + self.learning_input_list[i-2][1])

            x.append(x_avg)
            y.append(y_avg)

        plt.plot(x,y, label='inference result')


        x_des = []
        y_des = []
        #compile actual trajectory
        for state in self.learning_input_list:
            x_des.append(state[0])
            y_des.append(state[1])

        x_act = []
        y_act = []
        #compile actual trajectory
        for state in self.learning_state_list:
            x_act.append(state[0])
            y_act.append(state[1])


        #plot trajectory
        plt.plot(x_des, y_des, label='desired traj')
        plt.plot(x_act, y_act, label='actual traj')
        plt.legend()
        plt.show()



    def load_saved_model(self):

        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("model.h5")
        print("Loaded model from disk")
         
        # evaluate loaded model on test data
        self.model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['acc', 'mae'])
        
        

if __name__ == '__main__':
    # write code to create ROSControllerNode
    rospy.init_node('train_model')
    learner = Learner()


    x_train = np.load('x_train0.npy')
    y_train = np.load('y_train0.npy')

    x_t = np.load('x_train1.npy')
    y_t = np.load('y_train1.npy')

    x_train = np.concatenate((x_train,x_t), axis=0)
    y_train = np.concatenate((y_train,y_t), axis=0)

    x_t = np.load('x_train2.npy')
    y_t = np.load('y_train2.npy')

    x_train = np.concatenate((x_train,x_t), axis=0)
    y_train = np.concatenate((y_train,y_t), axis=0)

    for i in range(3):
        #reset lists 
        learner.learning_state_list = []
        learner.learning_input_list = []

        print('waiting for first msg')
        rospy.wait_for_message("/aer1217/learning_state", FlatState)
        print('Loading data')

        current_time = rospy.Time.now().to_sec() 
        while (current_time - learner.last_msg_time) < 5 :
            current_time =  rospy.Time.now().to_sec()
        x_t, y_t = learner.compile_training_data()

        np.save('x_train'+str(i)+'_reverse.npy', x_t)
        np.save('y_train'+str(i)+'_reverse.npy', y_t)

        # if i==0:
        #     x_train = x_t
        #     y_train = y_t
        # else:
        x_train = np.concatenate((x_train,x_t), axis=0)
        y_train = np.concatenate((y_train,y_t), axis=0)
        print('data_loaded')

    #Shuffle training data
    x_train, y_train = shuffle(x_train, y_train)
    x_train, y_train = learner.normalize_data(x_train, y_train)

    np.save('x_train.npy', x_train)
    np.save('y_train.npy', y_train)


    #All cases
    # learner.max_pos = 1.1148116815609779
    # learner.min_pos = -1.0464868543970438
    # learner.max_vel =  10.230594089786473
    # learner.min_vel =  -8.981004868385243
    # learner.max_acc =  25.535869053216068
    # learner.min_acc =  -26.926836052188715

    # #Cases 1, 2, 3, 4
    # learner.max_pos=  1.1148116815609779
    # learner.min_pos=  -0.6377376334306968
    # learner.max_vel=  10.230594089786473
    # learner.min_vel=  -5.94381835900806
    # learner.max_acc=  25.535869053216068
    # learner.min_acc=  -26.926836052188715

    # #cases 1,2,3
    # learner.max_pos=  1.1148116815609779
    # learner.min_pos=  -0.6377376334306968
    # learner.max_vel=  6.608598081092131
    # learner.min_vel=  -5.94381835900806
    # learner.max_acc=  5.3066485500455425
    # learner.min_acc=  -5.515697649409495


    # x_train = np.load('x_train.npy')
    # y_train = np.load('y_train.npy')


    print('Started training')
    history = learner.train_model(x_train, y_train)

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


    #print min max values used for training
    print('max_pos: ', learner.max_pos)
    print('min_pos: ', learner.min_pos)
    print('max_vel: ', learner.max_vel)
    print('min_vel: ', learner.min_vel)
    print('max_acc: ', learner.max_acc)
    print('min_acc: ', learner.min_acc)

    learner.inference()