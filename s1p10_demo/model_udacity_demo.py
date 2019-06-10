import os

import cv2 as cv

import numpy as np
from math import sin, cos, tan
from tqdm import tqdm

import matplotlib.pyplot as plt

from keras.models import model_from_json


def one_hot_to_angle(one_hot_arr):
    """
    Decode one hot encoded vector to get angle
    
    Input:
        one_hot_arr (np.ndarray): shape (1, num_classes)
        bins_edge (np.ndarray): shape (1, num_classes + 1)
    """
    angle_id = np.argmax(one_hot_arr)
    if angle_id < NUM_CLASSES:
        return (BINS_EDGE[angle_id] + BINS_EDGE[angle_id + 1]) / 2.
    else:
        return BINS_EDGE[angle_id]
    

def trans_2d(phi, s):
    """
    Generate 2D homogeneous transformation matrix given phi is the angle of rotation around z-axis
    and s the arc-length of the motion represented by the transformation
    
    Input
        phi (float): rotation angle in radian
        s (float): arc-length
    
    Return
        np.ndarray shape (3 x 2)
    """
    if abs(phi) < 1e-4:
        return np.array([[1, 0, s],
                         [0, 1, 0],
                         [0, 0, 1]])
    else:
        R = s / phi  # radius of rotation
        return np.array([[cos(phi), -sin(phi),  R * sin(phi)],
                         [sin(phi),  cos(phi),  R * (1 - cos(phi))],
                         [0,                0,  1]])
        

def steering_angle_to_way_pts(y_hat, s=2.0, L=3.7):
    """
    Convert an array of steering angles to an array of way points
    
    Input
        y_hat (np.ndarray): shape (1, NUM_LABELS)each element is a steering angle
        s (float): arc-length
        L (float): car length
        
    Output
        np.ndarray, shape(2, NUM_LABELS) each column is coordinate of car's tip i*2meter away 
        in frame attached to center of rear axel at the current iamge frame
    """

    # homogeneous coordinate of car's tip in any local frame
    iLi = np.array([[L, 0, 1]]).T 
    
    # initilize
    num_way_pts = len(y_hat)
    way_pts = np.zeros((3, num_way_pts))  # 3 rows, cuz of homogeneous coordinate
    oTi = np.eye(3)
    for i, angle in enumerate(y_hat):
        # calculate phi
        phi = s * tan(angle) / L
        # construct oTi
        oTi = oTi.dot(trans_2d(phi, s))
        # calculate oLi
        oLi = oTi.dot(iLi)
        # store oLi
        way_pts[:, i] = oLi.squeeze() 
        
    return y_hat, way_pts[:2, :]


if __name__ == "__main__":
    
    # Get image name
#     img_dir = "/home/user/Bureau/Dataset/udacity/CH2_002_output/center"
    img_dir = "/home/user/Bureau/Dataset/udacity/Ch2_001/center"
    img_list = []
    for (dirpath, dirnames, filenames) in os.walk(img_dir):
        img_list.extend(filenames)
        break
    img_list.sort()
    # Filter img_list
    filter_img_list = []
    for name in img_list:
        if '._' not in name:
            filter_img_list.append(name)
    
    # Neural net setup
    BINS_EDGE = np.load("./nn_data/s1p10_bins_edge.npy")
    NUM_LABELS = 10
    NUM_CLASSES = len(BINS_EDGE) - 1
    
    NUM_PRED = 4
    NUM_SMOOTH = 5
    NUM_DEMO = 500
    
    prediction_file = "./nn_data/prediction_CH2_001.npy"
        
    if not os.path.isfile(prediction_file):   
        print("[INFO] Prediction does not exist")
        print("[INFO] Load model")
        # Load neural net model
        MODEL_DIR = "./nn_data/"
        MODEL_NAME = "ext_bottom_half_s1p10_model_2019_06_09_17_38"

        with open(MODEL_DIR + "%s.json" % MODEL_NAME, 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)

        model.load_weights(MODEL_DIR + "%s.h5" % MODEL_NAME)
        print("[INFO] Loaded Neural Net from disk")
        print("[INFO] Compute prediction")
        # Precompute steering angle
        X = np.zeros((NUM_DEMO, 200, 200, 1))
        for i, name in enumerate(filter_img_list):
            if i >= NUM_DEMO:
                break
            filename = os.path.join(img_dir, name)
            _gray_img = cv.imread(filename, 0)

            # get bottom half
            bottom_half = _gray_img[int(0.5 * _gray_img.shape[0]) :, :]

            # down sample & reshape image
            __img = np.float32(cv.resize(bottom_half, (200, 200), interpolation=cv.INTER_AREA))
            if len(__img.shape) == 2:
                __img = __img.reshape((200, 200, 1))

            # store image to X
            X[i, :, :, :] = __img
        y_hat_list = model.predict(X, verbose=1)
        np.save(prediction_file, y_hat_list)
    else:
        print("[INFO] Load prediction")
        y_hat_tensor = np.load(prediction_file)
        y_hat_list = [y_hat_tensor[lab_idx, :, :] for lab_idx in range(NUM_LABELS)]
    
    print("[INFO] Prediction shape: ", y_hat_list[0].shape)
    
    # Display image and prediction in main loop
    f = plt.figure(figsize=(10,3))
    
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    
    for i, img_name in enumerate(filter_img_list):
        if i >= NUM_DEMO - NUM_SMOOTH:
            print("Stop for debugging purpose")
            break
            
        # read image
        img_path = os.path.join(img_dir, img_name)
        img = cv.imread(img_path, 0)
        
        # get prediction of steering angle
        y_hat_mat = np.zeros((NUM_SMOOTH, NUM_PRED))
        for i_off in range(NUM_SMOOTH):
            y_hat_enc = [y_hat_list[lab_idx][i + i_off, :] for lab_idx in range(NUM_PRED)]
            # Decode pred_y to get a sequence of steering angle
            y_hat_mat[i_off, :] = np.array([one_hot_to_angle(y_hat_enc[lab_idx]) for lab_idx in range(NUM_PRED)])
        
        # average each column of y_hat_mat 
        y_hat = np.average(y_hat_mat, axis=0)
        
        # get way points for the tip of the car in frame attached to center of rear axel
        pred_sequence, way_pts = steering_angle_to_way_pts(y_hat, L=1.5)
        print("[INFO] predit sequence: ", pred_sequence)
        # display
        ax1.clear()
        ax2.clear()
        
        ax1.set_title("Perspective view")
        ax1.imshow(img, cmap='gray')
        
        ax2.set_title("[World] Bird-eye view")
        ax2.set_xlabel("Lateral (m)")
        ax2.set_ylabel("Longitual (m)")
        ax2.set_xlim(2., -2.)
        ax2.set_ylim(0., 25.)
        ax2.plot(way_pts[1, :], way_pts[0, :], 'r:')
        
        plt.pause(.15)
        plt.draw()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    