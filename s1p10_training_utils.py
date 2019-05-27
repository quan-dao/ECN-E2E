import keras
import pickle
import pandas as pd
import numpy as np
import cv2 as cv


def save_lstm(lstm_obj, time_str, weight_dir):
    lstm_w = lstm_obj.get_weights()
    lstm_w_dict ={}
    lstm_w_dict['0'] = lstm_w[0]
    lstm_w_dict['1'] = lstm_w[1]
    lstm_w_dict['2'] = lstm_w[2]

    with open(weight_dir + "/lstm_weights_%s.p" % time_str, 'wb') as fp:
        pickle.dump(lstm_w_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

        
def one_hot_encode(angle_id, num_class):
    """
    Convert angle id to one-hot encoding vector
    """
    y = np.zeros(num_class)
    y[angle_id] = 1
    return y


class DataGenerator(keras.utils.Sequence):
    def __init__(self, 
                 dataset_csv, data_root_dir, 
                 img_shape, 
                 num_class, 
                 num_prediction, 
                 batch_size=30, 
                 shuffle=True, 
                 lstm_dim_hidden_states=None):
        """
        Input:
            dataset_csv (string): path to csv contains dataset
            data_root_dir (str): path to root folder of all dataset
            img_shape (tuple): (img_height, img_width, img_channel)
            num_class (int): number of classes of angle
            num_prediction (int): number of angles will be predicted 
            batch_size (int): size of a training batch
            shuffle (bool): shuffle dataset after 1 epoch
        """
        self.df = pd.read_csv(dataset_csv)
        self.data_root_dir = data_root_dir
        
        # check channel of image
        if img_shape[-1] == 1:
            self.color_img = False
        else:
            self.color_img = True
            
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # for training a classifier 
        self.num_prediction = num_prediction
        self.num_class = num_class
        
        # for LSTM
        self.lstm_dim_hidden_states = lstm_dim_hidden_states
        
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
            X (np.ndarray): shape of (batch_size, img_height, img_width, img_channel) 
            y (np.ndarray): label vector, shape (batch_size, num_prediction)
        """
        
        X = np.zeros((self.batch_size, ) + self.img_shape)
        y = [np.zeros((self.batch_size, self.num_class)) for i in range(self.num_prediction)]
        
        # Iterate through each idx in training batch
        for i, idx in enumerate(list_indexes): 
            # get image file name
            file_name = self.df.iloc[idx].frame_name
            # read image
            if not self.color_img:
                img = cv.imread(self.data_root_dir + file_name, 0)
            else:
                img = cv.imread(self.data_root_dir + file_name, 1)
            # resize & reshape image
            img = np.float32(cv.resize(img, 
                                       (self.img_shape[1], self.img_shape[0]), 
                                       interpolation=cv.INTER_AREA))
            if len(img.shape) == 2:
                img = img.reshape((self.img_shape))
            
            # store image to X
            X[i, :, :, :] = img
            
            # store angle to y
            angle_id_list = self.df.iloc[idx].angle_id[1: -1].split(", ")
            for j, angle_id in enumerate(angle_id_list):
                y[j][i, :] = one_hot_encode(int(angle_id), self.num_class)
        
        # create additional input for LSTM layer
        if self.lstm_dim_hidden_states:
            a0 = np.zeros((self.batch_size, self.lstm_dim_hidden_states))
            c0 = np.zeros((self.batch_size, self.lstm_dim_hidden_states))
            y0 = np.zeros((self.batch_size, self.num_class))
            return [X, a0, c0, y0], y
        else:
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