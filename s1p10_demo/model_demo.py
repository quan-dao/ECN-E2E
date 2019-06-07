import os

import cv2 as cv

import numpy as np
from math import sin, cos, tan

import matplotlib.pyplot as plt

from image import load_image
from camera_model import CameraModel

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
    way_pts = np.zeros((3, NUM_LABELS))  # 3 rows, cuz of homogeneous coordinate
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


def put_in_cam(way_pts, cTw):
    """
    Transfrom an array of way points in world frame (frame attached to center of rear axel) into 
    camera frame (camera set up is according to Oxford robotcar)
    """
    cam_way_pts = np.zeros((3, way_pts.shape[1]))
    for i in range(way_pts.shape[1]):
        world_pt = np.array([[way_pts[0, i], way_pts[1, i], 0, 1]])
        cam_way_pts[:, i] = cTw.dot(world_pt.T).squeeze()
    return cam_way_pts


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


if __name__ == "__main__":
    
    # Get image name
    img_dir = "/home/user/Downloads/sample_large/stereo/centre"
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
    
    # Load camera model
    cam_model_dir = "./extrinsics"
    cam_model = CameraModel(cam_model_dir, img_dir)
    
    
    if not os.path.isfile("./nn_data/prediction_oxford_large.npy"):   
        print("[INFO] Prediction does not exist")
        print("[INFO] Load model")
         # Neural net setup
        BINS_EDGE = np.load("./nn_data/s1p10_bins_edge.npy")
        NUM_LABELS = 10
        NUM_CLASSES = len(BINS_EDGE) - 1
        # Load neural net model
        MODEL_DIR = "./nn_data/"
        MODEL_NAME = "ext_bottom_half_s1p10_model_2019_06_07_14_01"

        with open(MODEL_DIR + "%s.json" % MODEL_NAME, 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)

        model.load_weights(MODEL_DIR + "%s.h5" % MODEL_NAME)
        print("[INFO] Loaded Neural Net from disk")
        print("[INFO] Compute prediction")
        # Precompute steering angle
        X = np.zeros((len(filter_img_list), 200, 200, 1))
        for i, name in enumerate(filter_img_list):
            filename = os.path.join(img_dir, name)
            _img = load_image(filename, cam_model)
            _img = np.round(_img).astype(int)

            # convert to gray scale
            _gray_img = rgb2gray(_img)

            # get bottom half
            bottom_half = _gray_img[int(0.5 * _gray_img.shape[0]) :, :]

            # down sample & reshape image
            __img = np.float32(cv.resize(bottom_half, (200, 200), interpolation=cv.INTER_AREA))
            if len(__img.shape) == 2:
                __img = __img.reshape((200, 200, 1))

            # store image to X
            X[i, :, :, :] = __img

        y_hat_list = model.predict(X, verbose=1)
        np.save("./nn_data/prediction_oxford_large.npy", y_hat_list)
    else:
        print("[INFO] Load prediction")
        y_hat_tensor = np.load("./nn_data/prediction_CH2_001.npy")
        y_hat_list = [y_hat_tensor[lab_idx, :, :] for lab_idx in range(NUM_LABELS)]
    
    print("[INFO] Prediction shape: ", y_hat_list[0].shape)

    # pose of world frame (frame attached to center of rear axel) w.r.t camera frame
    cTw = np.array([[1,  0,  0, -1.72],
                    [0, -1,  0, 0.12],
                    [0,  0, -1, 1.2]])
    
    # Display image and prediction in main loop
    f = plt.figure(figsize=(10,3))
    
    ax = f.add_subplot(111)
    ax.set_title("Perspective view")
    
    for i, img_name in enumerate(filter_img_list):
        # read image
        img_path = os.path.join(img_dir, img_name)
        img = load_image(img_path, cam_model)
        img = np.round(img).astype(int)
        # convert to gray scale
        gray_img = rgb2gray(img)
        
        # get prediction of steering angle
        y_hat_enc = [y_hat_list[lab_idx][i, :] for lab_idx in range(NUM_LABELS)]
        # Decode pred_y to get a sequence of steering angle
        y_hat = [one_hot_to_angle(y_hat_enc[lab_idx]) for lab_idx in range(NUM_LABELS)]
                
        # get way points for the tip of the car in frame attached to center of rear axel
        pred_sequence, way_pts = steering_angle_to_way_pts(y_hat)
        
        # transform way points in world frame to camera frame
        cam_way_pts = put_in_cam(way_pts, cTw)
        
        # project way points onto image plane
        uv, _ = cam_model.project(cam_way_pts, img.shape)
        uv = np.round(uv).astype(int)
        
        # draw projected way points
        for i in range(uv.shape[1]-1):
            p1 = (uv[0, i], uv[1, i]) 
            p2 = (uv[0, i + 1], uv[1, i + 1])
            cv.line(gray_img, p1, p2, (0, 255, 0), 10)
        
        # display
        ax.clear()
        ax.imshow(gray_img, cmap='gray')
        
        plt.pause(.15)
        plt.draw()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    