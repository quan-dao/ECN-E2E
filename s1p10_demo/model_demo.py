import os

import cv2 as cv

import numpy as np
from math import sin, cos, tan

import matplotlib.pyplot as plt

from image import load_image
from camera_model import CameraModel


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
        y_hat (list): each element is np.ndarray shape (1, NUM_LABELS)
        s (float): arc-length
        L (float): car length
        
    Output
        np.ndarray, shape(2, NUM_LABELS) each column is coordinate of car's tip i*2meter away 
        in frame attached to center of rear axel at the current iamge frame
    """

    # Decode pred_y to get a sequence of steering angle
    pred_sequence = [one_hot_to_angle(y_hat[i]) for i in range(NUM_LABELS)]
    
    # homogeneous coordinate of car's tip in any local frame
    iLi = np.array([[L, 0, 1]]).T 
    
    # initilize
    way_pts = np.zeros((3, NUM_LABELS))  # 3 rows, cuz of homogeneous coordinate
    oTi = np.eye(3)
    for i, angle in enumerate(pred_sequence):
        # calculate phi
        phi = s * tan(angle) / L
        # construct oTi
        oTi = oTi.dot(trans_2d(phi, s))
        # calculate oLi
        oLi = oTi.dot(iLi)
        # store oLi
        way_pts[:, i] = oLi.squeeze() 
        
    return pred_sequence, way_pts[:2, :]


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
    
    BINS_EDGE = np.load("./nn_data/s1p10_bins_edge.npy")
    NUM_LABELS = 10
    NUM_CLASSES = len(BINS_EDGE) - 1
    
    # Get model prediction (kind of cheating to accelerate model demotration)
    whole_y_hat = np.load("small_sample_prediction.npy")
    y_hat_list = [whole_y_hat[i, :, :]for i in range(NUM_LABELS)] 

    print("Prediction shape: ", y_hat_list[0].shape)
    
    # Get image name
    img_dir = "/home/user/Downloads/sample_small/stereo/centre"
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
    
    # pose of world frame (frame attached to center of rear axel) w.r.t camera frame
    cTw = np.array([[1,  0,  0, -1.72],
                    [0, -1,  0, 0.12],
                    [0,  0, -1, 1.2]])
    
    # Display image and prediction in main loop
    f = plt.figure(figsize=(10,3))
    
    ax = f.add_subplot(131)
    ax.set_title("Perspective view")
    
    ax2 = f.add_subplot(132)
#     ax2.set_title("[World] Bird-eye view")
#     ax2.set_xlabel("Lateral (m)")
#     ax2.set_ylabel("Longitual (m)")
#     ax2.set_xlim(2., -2.)
#     ax2.set_ylim(0., 25.)
    
    ax3 = f.add_subplot(133)
#     ax3.set_title("[Camera] Bird-eye view")
#     ax3.set_xlabel("Lateral (m)")
#     ax3.set_ylabel("Longitual (m)")
#     ax3.set_xlim(-2., 2.)
#     ax3.set_ylim(0., 25.)
    
    for i, img_name in enumerate(filter_img_list):
        # read image
        img_path = os.path.join(img_dir, img_name)
        img = load_image(img_path, cam_model)
        img = np.round(img).astype(int)
        # convert to gray scale
        gray_img = rgb2gray(img)
        
        # get prediction of steering angle
        y_hat = [y_hat_list[j][i, :] for j in range(NUM_LABELS)]
        
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
        ax2.clear()
        ax3.clear()

        ax2.set_title("[World] Bird-eye view")
        ax2.set_xlabel("Lateral (m)")
        ax2.set_ylabel("Longitual (m)")
        ax2.set_xlim(2., -2.)
        ax2.set_ylim(0., 25.)

        ax3.set_title("[Camera] Bird-eye view")
        ax3.set_xlabel("Lateral (m)")
        ax3.set_ylabel("Longitual (m)")
        ax3.set_xlim(-2., 2.)
        ax3.set_ylim(0., 25.)
        ax.imshow(gray_img, cmap='gray')
        ax2.plot(way_pts[1, :], way_pts[0, :], 'r:')
        ax3.plot(cam_way_pts[1, :], cam_way_pts[0, :], 'r:')
        
        plt.pause(.1)
        plt.draw()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    