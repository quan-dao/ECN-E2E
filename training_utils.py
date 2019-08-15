import pickle
import pandas as pd
import numpy as np
import cv2 as cv
from scipy.stats import bernoulli
from tqdm import tqdm


# ----------------------Data Helper -------------------------------------            
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
            
            
def gen_classifier_dataset(df_path, 
                           num_classes, num_labels, bins_edge, 
                           image_shape, 
                           num_samples=None, data_root_dir=None, flip_prob=0.5):
    """
    Generate dataset as list of numpy array from dataframe
    
    Input:
        df (pandas.DataFrame)
        num_labels (int)
        image_shape (tuple)
       
    Output:
        X (numpy.ndarray): model has 1 input
        y (list) of numpy.ndarray (y is a list because model has multiple outputs)
    
    """
    df = pd.read_csv(df_path)
    
    if not num_samples:
        num_samples = min(len(df), 19000)
    
    X = np.zeros((num_samples, ) + image_shape) # only 1 input
    
    # Note: j is used to iterate output list
    y = [np.zeros((num_samples, num_classes)) for j in range(num_labels)]
    
    flip = bernoulli.rvs(flip_prob, size=num_samples) == 1
    
    # Iterate through the whole dataset
    for i in tqdm(range(num_samples)):  # i is used to iterate batch 
        # get image file name
        file_name = df.iloc[i].frame_name
        # read image
        img = cv.imread(data_root_dir + file_name, 0)
        
        bottom_half = img[100 : , :]
        
        # down sample & reshape image
        img = np.float32(cv.resize(bottom_half, (image_shape[1], image_shape[0]), interpolation=cv.INTER_AREA))
        if len(img.shape) == 2:
            img = img.reshape((image_shape))

        # check if this sample needs to be flipped
        if flip[i]:
            img = np.fliplr(img)

        # store image to X
        X[i, :, :, :] = img
        
        # create y
        angle_val_list = df.iloc[i].angle_val[1: -1].split(", ")
        
        for j, angle_val in enumerate(angle_val_list):
            # find angle_id
            if flip[i]:
                angle_id = find_angle_id(-float(angle_val), bins_edge)
            else:
                angle_id = find_angle_id(float(angle_val), bins_edge)
            # one hot encode
            y[j][i, :] = one_hot_encode(angle_id, num_classes)

    return X, y
         

def gen_regressor_dataset(df_path, 
                           num_labels,  
                           image_shape, 
                           num_samples=None, data_root_dir=None, flip_prob=0.5):
    """
    Generate dataset as list of numpy array from dataframe
    
    Input:
        df (pandas.DataFrame)
        num_labels (int)
        image_shape (tuple)
       
    Output:
        X (numpy.ndarray): model has 1 input
        y (np.ndarray) - (num_samples, num_labels)
    
    """
    df = pd.read_csv(df_path)
    
    if not num_samples:
        num_samples = min(len(df), 19000)
    
    X = np.zeros((num_samples, ) + image_shape) # only 1 input
    y = np.zeros((num_samples, num_labels))
    
    flip = bernoulli.rvs(flip_prob, size=num_samples) == 1
    
    # Iterate through the whole dataset
    for i in tqdm(range(num_samples)):  # i is used to iterate batch 
        # get image file name
        file_name = df.iloc[i].frame_name
        # read image
        img = cv.imread(data_root_dir + file_name, 0)
        
        bottom_half = img[100 : , :]
        
        # down sample & reshape image
        img = np.float32(cv.resize(bottom_half, (image_shape[1], image_shape[0]), interpolation=cv.INTER_AREA))
        if len(img.shape) == 2:
            img = img.reshape((image_shape))

        # check if this sample needs to be flipped
        if flip[i]:
            img = np.fliplr(img)

        # store image to X
        X[i, :, :, :] = img
        
        # create y
        angle_val_list = df.iloc[i].angle_val[1: -1].split(", ")
        if flip[i]:
            y[i, :] = np.array([-float(angle) for angle in angle_val_list])
        else:
            y[i, :] = np.array([float(angle) for angle in angle_val_list])

    return X, y
 