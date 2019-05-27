import keras
import pickle
import pandas as pd
import numpy as np
import cv2 as cv
from scipy.stats import bernoulli


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


def find_angle_id(angle, bin_edges):
    """
    Find id for idx-th row of center_cam_df w.r.t this row's steering angle
    
    Input:
        idx (scalar): row index of center_cam_df 
        df (pd.Dataframe): dataframe  
        bin_edges (np.array): array of bin edges 
    
    Output:
        ID of this angle
    """
    angle_ID = -1
    i = 0
    flag_found_bin = False
    while i < len(bin_edges) - 1 and not flag_found_bin:
        if angle >= bin_edges[i] and angle < bin_edges[i + 1]:
            flag_found_bin = True
            angle_ID = i  
        else:
            i += 1
    
    if not flag_found_bin:
        # not found any bin contains this steering angle --> equal to the right edge of the last bin
        angle_ID = i - 1
    
    return angle_ID


class DataGenerator(keras.utils.Sequence):
    def __init__(self, 
                 dataset_csv, data_root_dir, 
                 img_shape, 
                 num_class, 
                 num_prediction,
                 bins_edge,
                 batch_size=30, 
                 shuffle=True, 
                 lstm_dim_hidden_states=None,
                 flip_prob=0.5):
        """
        Input:
            dataset_csv (string): path to csv contains dataset
            data_root_dir (str): path to root folder of all dataset
            img_shape (tuple): (img_height, img_width, img_channel)
            num_class (int): number of classes of angle
            num_prediction (int): number of angles will be predicted 
            bins_edge (list): list of outter edge of every bin
            batch_size (int): size of a training batch
            shuffle (bool): shuffle dataset after 1 epoch
            lstm_dim_hidden_states (int): dimension of hidden states of LSTM cell
            flip_prob (float): probability of flipping image & subsequece steering angles
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
        self.flip_prob = flip_prob
        self.bins_edge = bins_edge
        
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
        
        # decide when to flip the image
        flip = bernoulli.rvs(self.flip_prob, size=self.batch_size) == 1
        
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
            
            # flip the image if any
            if flip[i]:
                img = np.fliplr(img)
            
            # store image to X
            X[i, :, :, :] = img
            
            # create y
            angle_val_list = self.df.iloc[idx].angle_val[1: -1].split(", ")
            for j, angle_val in enumerate(angle_val_list):
                if flip[i]:
                    # flip image -> flip angle
                    angle_id = find_angle_id(-float(angle_val), self.bins_edge)
                else:
                    # No flip
                    angle_id = find_angle_id(float(angle_val), self.bins_edge)
                # one-hot encode angle_id & store result to y
                y[j][i, :] = one_hot_encode(angle_id, self.num_class)
        
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