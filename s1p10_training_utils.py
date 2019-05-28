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

        
def save_layer_weight(layer, time_str, weight_dir, name):
    weight = layer.get_weights()
    w_dict ={}
    for i, w in enumerate(weight):
        w_dict[i] = w
        
    with open(weight_dir + "/%s_weights_%s.p" % (name, time_str), 'wb') as fp:
        pickle.dump(w_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
        
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
                 dataset_csv, 
                 num_class, 
                 num_labels,
                 bins_edge,
                 batch_size=30, 
                 shuffle=True, 
                 lstm_dim_hidden_states=None,
                 flip_prob=0.5):
        """
        Input:
            dataset_csv (string): path to csv contains dataset
            num_class (int): number of classes of angle
            num_prediction (int): number of angles will be predicted 
            bins_edge (list): list of outter edge of every bin
            batch_size (int): size of a training batch
            shuffle (bool): shuffle dataset after 1 epoch
            lstm_dim_hidden_states (int): dimension of hidden states of LSTM cell
            flip_prob (float): probability of flipping image & subsequece steering angles
        """
        self.df = pd.read_csv(dataset_csv)
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # for training a classifier 
        self.num_labels = num_labels
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
        # Initialize X & y
        y = [np.zeros((self.batch_size, self.num_class)) for i in range(self.num_labels - 1)]
        X = [np.zeros((self.batch_size, self.num_class)) for i in range(self.num_labels - 1)]
        
        # decide when to flip the image
        flip = bernoulli.rvs(self.flip_prob, size=self.batch_size) == 1
        
        # Iterate through each idx in training batch
        for i, idx in enumerate(list_indexes): 
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
                one_hot_vec = one_hot_encode(angle_id, self.num_class) 
#                 y[j][i, :] = one_hot_encode(angle_id, self.num_class)
                
                # create X by shifting y to the past (the left)
                if j == 0:
                    X[j][i, :] = one_hot_vec
                elif j < self.num_labels - 1:
                    y[j - 1][i, :] = one_hot_vec
                    X[j][i, :] = one_hot_vec
                else:
                    y[j - 1][i, :] = one_hot_vec
                    
        # define a_0 & c_0
        a_0 = np.zeros((self.batch_size, self.lstm_dim_hidden_states))
        c_0 = np.zeros((self.batch_size, self.lstm_dim_hidden_states))
        
        return X + [a_0, c_0], y
        
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