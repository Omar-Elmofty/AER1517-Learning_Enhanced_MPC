import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

class Learner(object):
    def __init__(self):

        #normalization variables
        self.max_pos = 0
        self.min_pos = 0
        self.max_vel = 0
        self.min_vel = 0
        self.max_acc = 0
        self.min_acc = 0

        #Keras model
        self.model = None


    def normalize_data(self, x_train, y_train):
        """Function that normalizes training data
        Args:
            x_train: training data
            y_train: training labels
        Returns: normalized x_train, y_train
        """
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


    def create_model(self):
        """Function for defining the DNN model
        """
        #model architecture
        model = Sequential()
        model.add(Dense(12, input_shape = (12,), activation = 'relu'))
        model.add(Dense(10, activation = 'relu'))
        model.add(Dense(6))
        #compile model
        sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['acc', 'mae'])

        #print model summary
        model.summary()

        return model

    def train_model(self, x_train, y_train):
        """Function for training the DNN
        Returns: history of training
        """

        #create model
        self.model = self.create_model()
        
        #train model
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
        
        

if __name__ == '__main__':
    
    learner = Learner()

    #Load training data
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')


    #Shuffle training data
    x_train, y_train = shuffle(x_train, y_train)
    #normalize data
    x_train, y_train = learner.normalize_data(x_train, y_train)

    #Train model
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
    print('Normalization coefficients')
    print('max_pos: ', learner.max_pos)
    print('min_pos: ', learner.min_pos)
    print('max_vel: ', learner.max_vel)
    print('min_vel: ', learner.min_vel)
    print('max_acc: ', learner.max_acc)
    print('min_acc: ', learner.min_acc)