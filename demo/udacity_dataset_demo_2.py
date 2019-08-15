import numpy as np
from math import cos, sin, tan
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

L = 2.3  # rear-to-front axel distance

num_middle_way_pts = 20
arc_lens = [2./(num_middle_way_pts - 1) for i in range(num_middle_way_pts)]

X_val = np.load("../data/demo_X.npy")
y_val = np.load("../data/demo_y.npy")

if not os.path.isfile("../data/demo_y_pred.npy"):
    # Generate y prediction
    from keras.models import load_model
    model = load_model("../best_weights/s1p5_2019_07_31_09_24.h5")
    y_pred = model.predict(X_val)
    np.save("../data/demo_y_pred.npy", y_pred)
else:
    print("Load y_pred")
    y_pred = np.load("../data/demo_y_pred.npy")

bins_edge = np.load("../data/bins_edge.npy")

def next_pose(angle, dist):
    """
    Calculate next pose w.r.t the current frame given the steering angle
    and the travelled distance
    Input:
        angle - scalar: steering angle applied to the whole travelled distance
        dist - scalar: travelled distance
    Output:
        Trans - (3, 3): pose of next frame
    """
    if abs(angle) < 1e-4:
        # vehicle moves in a straight line
        Trans = np.eye(3)
        Trans[0, 2] = dist  # new position (along the current x-axis)
    else:
        # vehicle moves along an arc
        radius = L / tan(angle)
        phi = dist / radius
        # new position
        t_x = 2 * radius * sin(phi/2) * cos(phi/2)
        t_y = 2 * radius * sin(phi/2) * sin(phi/2)
        # new pose
        Trans = np.array([[cos(phi), -sin(phi), t_x],
                          [sin(phi), cos(phi), t_y],
                          [0, 0, 1]])
    return Trans


def steering_seq_to_waypts(steering):
    """
    Convert a sequence of steering angles into waypoints
    Input:
        steering - (1, num_steerings)
    Output:
        way_pts - (2, num_points)
    """
    T_0_i = np.eye(3)
    way_pts = []
    for angle in steering:
        for dist in arc_lens:
            T_0_i = T_0_i @ next_pose(angle, dist)
            way_pts.append(T_0_i[:2, 2])
    way_pts = np.vstack(way_pts)
    return way_pts


def one_hot_to_angle(one_hot):
    """
    Convert one-hot vector to angle
    Input:
        one_hot - (1, num_classes): one-hot encoded vector
    Output:
        angle - scaler: mean value of the corresponding bin
    """
    idx = np.argmax(one_hot)
    return (bins_edge[idx] + bins_edge[idx + 1])/2


def img_to_waypts(img_idx, y):
    """
    Predict sequence of waypoints given an image
    Input:
        img_idx - scalar: index in X_val of input image
        y - list - (batch_size, num_classes): either y_pred or y_val
    Output:
        way_pts - (2, num_points) 
    """
    one_hot = [y[i][img_idx, :] for i in range(5)]
    steering = [one_hot_to_angle(vector) for vector in one_hot]
    way_pts = steering_seq_to_waypts(steering)
    return steering, way_pts


f, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, constrained_layout=True)
# f.tight_layout()

for i in range(X_val.shape[0]):
    # Get image

    img = X_val[i, ...].squeeze()

    # Generate sequence of way points
    true_steering, true_way_pts = img_to_waypts(i, y_val)
    pred_steering, pred_way_pts = img_to_waypts(i, y_pred)
    print("True steering:\n", true_steering)
    print("Predicted steering:\n", pred_steering)

    # display
    ax1.clear()
    ax1.set_title("Perspective view")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(img, cmap='gray')

    ax2.clear()
    ax2.set_title("[Bird-eye] Way points")
    ax2.set_xlabel("Lateral (m)")
    ax2.set_ylabel("Longitudal (m)")
    ax2.set_xlim(-3., 3.)
    ax2.invert_xaxis()
    ax2.set_ylim(0., 12.)
    ax2.plot(true_way_pts[:, 1], true_way_pts[:, 0], 'r-.', label='ground truth')
    ax2.plot(pred_way_pts[:, 1], pred_way_pts[:, 0], 'b-.', label='predicted')
    ax2.legend(loc="upper right")


    plt.pause(.15)
    plt.draw()

    if i > 1000:
        break
