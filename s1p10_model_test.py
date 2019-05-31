from os import walk
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


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
    

def steering_angle_to_next_pos(angle, arc_length=2.0):
    """
    Calculate next position in the body frame of the current image frame
        Input:
            angle (float): steering angle in radian
    """
    if np.absolute(angle) < 1e-5:
        next_y = 0
        next_x = arc_length
    else:
        R = arc_length / angle
        next_y = R * (1 - np.cos(angle))
        next_x = R * np.sin(angle)
    return np.array([next_x, next_y])


def steering_angle_to_way_pts(y_hat):
    """
    Convert an array of steering angles to an array of way points
    
    Input
        y_hat (list): each element is np.ndarray shape (1, NUM_LABELS)
        
    Output
        np.ndarray, shape(6, 2) (1st column is x, 2nd column is y)
    """

    # Decode pred_y to get a sequence of steering angle
    pred_sequence = [one_hot_to_angle(y_hat[i]) for i in range(NUM_LABELS)]

    # initialize waypts
    way_pts = np.zeros((6, 2))
    for i, angle in enumerate(pred_sequence):
        next_pos = steering_angle_to_next_pos(angle)
        way_pts[i + 1, :] = way_pts[i, :] + next_pos 
        
    return pred_sequence, way_pts


# Get images name
IMG_DIR = "/home/user/Bureau/Dataset/udacity/Ch2_001/center/"
img_list = []
for (dirpath, dirnames, filenames) in walk(IMG_DIR):
    img_list.extend(filenames)
    break
img_list.sort()


# Get BINS_EDGE
BINS_EDGE = np.load("./s1p10_data/s1p10_bins_edge.npy")

# Get model prediction (kind of cheating to accelerate model demotration)
NUM_LABELS = 5
NUM_CLASSES = len(BINS_EDGE) - 1
whole_y_hat = np.load("./s1p10_data/CH2_001_s1p10_y_hat.npy")
whole_y_hat = [whole_y_hat[i, :, :]for i in range(NUM_LABELS)] 

print(whole_y_hat[0].shape)
# -------- Test Model--------------

f = plt.figure(figsize=(10,3))
ax = f.add_subplot(121)
ax2 = f.add_subplot(122)

for i in range(100):
    img_name = img_list[i]

    # Read image
    img = cv.imread(IMG_DIR + img_name, 0)
    
    # Get steering angle & way_pts
    y_hat = [whole_y_hat[j][i, :] for j in range(NUM_LABELS)]
    pred_sequence, way_pts = steering_angle_to_way_pts(y_hat)

    # Display
    ax.clear()
    ax.imshow(img, cmap='gray')
    ax2.clear()
    ax2.plot(way_pts[:, 1], way_pts[:, 0], 'r:')
    ax2.set_xlim(-2., 2.)
    
    plt.pause(.1)
    plt.draw()