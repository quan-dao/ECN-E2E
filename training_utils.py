import keras
import keras.backend as K
import pandas as pd
import cv2 as cv
import numpy as np


#---------------- HELPER FUNCTION -------------------------

# Define root mean square loss
def root_mean_squared_error(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true)))


def save_model(model, name):
    # serialize model to JSON
    #  the keras model which is trained is defined as 'model' in this example
    model_json = model.to_json()


    with open("./model/%s.json" % name, "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("./model/%s.h5" % name)


#---------------- HELPER CLASS ---------------------------
# DATA GENERATOR
# this class is used by model.fit_generator

class DataGenerator(keras.utils.Sequence):
    def __init__(self, dataset_csv, img_shape, batch_size=1, shuffle=True, additional_input_for_LSTM=False, LSTM_nb_hidden_states=None):
        """
        Input:
            dataset_csv (string): path to csv contains dataset
            img_shape (tuple): (img_height, img_width, img_channel)
            batch_size (int): size of a training batch
            shuffle (bool): shuffle dataset after 1 epoch
        """
        self.df = pd.read_csv(dataset_csv)
        
        # check channel of image
        if img_shape[-1] == 1:
            self.color_img = False
        else:
            self.color_img = True

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_shape = img_shape
        self.additional_input = additional_input_for_LSTM
        self.LSTM_nb_hidden_states = LSTM_nb_hidden_states
        # invoke on_epoch_end to create shuffle training dataset
        self.on_epoch_end()
    
    def __len__(self):
        """
        Output:
            the number of batches per epoch
        """
        return int(np.floor(len(self.df) / self.batch_size))
    
    def __data_generation(self, list_indexes):
        """
        Input:
            list_indexes (list): list of indexes of training sample in this batch
        
        Output:
            X (list): list of np.ndarray. Each have shape of (batch_size, img_height, img_width, img_channel) 
            y (np.ndarray): label vector, shape (batch_size, 25)
        """
        
        X = [np.zeros((self.batch_size,) + self.img_shape) for i in range(3)]
        
        y = np.zeros((self.batch_size, 25))
        
        img_path_prefix = '/home/user/Bureau/Dataset/udacity/'
        
        # Iterate through each idx in training batch
        for i, idx in enumerate(list_indexes): 
            file_names_list = self.df.iloc[idx].X[2: -2].split("', '")
            # read img & put it in  X_0, X_1, X_2
            for j, name in enumerate(file_names_list):
                # read image
                if not self.color_img:
                    img = cv.imread(img_path_prefix + name, 0)
                else:
                    img = cv.imread(img_path_prefix + name, 1)
                
                # resize & reshape image
                img = np.float32(cv.resize(img, (self.img_shape[1], self.img_shape[0]), 
                                           interpolation=cv.INTER_AREA))
                if len(img.shape) == 2:
                    img = img.reshape((self.img_shape))
                
                # add img to input tensor
                X[j][i, :, :, :] = img
            
            # get label
            y[i, :] = np.array([float(angle) for angle in self.df.iloc[idx].steering_angles[1: -1].split(", ")])
        
        if self.additional_input:
            x0 = np.zeros((self.batch_size, 1, 1))
            c0 = np.zeros((self.batch_size, self.LSTM_nb_hidden_states))
            return X + [x0, c0], y
        
        return X, y
    
    def __getitem__(self, index):
        """
        Generate one batch of data
        
        Input:
            index (int): index of the first training sample
        """
        # Generate indexes of the batch
        list_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        # if (index + 1) * self.batch_size > len(self.indexes), 
        # list_indexes = [index * self.batch_size: len(self.indexes)]

        # Generate data
        X, y = self.__data_generation(list_indexes)

        return X, y
    
    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.df))  # array of indexes of training dataset
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
                    

