import pandas as pd
import numpy as np
from math import pi, sin, cos


def put_in_PI(angle):
    """
    Put the input angle in the range [-pi, pi]
    
    Input:
        angle (float)
    Output:
        new_angle (float)
    """
    m = angle % (2*pi)
    if m > pi:
        new_angle = m - 2*pi
    elif m < -pi:
        new_angle = m + 2*pi
    else:
        new_angle = m
    
    return new_angle


def clamp_angle(angle, limit=30*pi/180, debug=True):
    """
    Clamp angle with the limit defined by "limit"
    Input:
        angle (float): rad
    """
    _angle = put_in_PI(angle)
    if _angle < -limit:
        if debug:
            print("Clamp %.4f" % _angle)
        return -limit
    elif _angle > limit:
        if debug:
            print("Clamp %.4f" % _angle)
        return limit
    else:
        return _angle
    
    
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
    
    
# HELPER FUNCTION FOR FINDING DISTANCE BETWEEN 2 FRAME

def lat_long_to_XYZ(lat, long):
    """
    Convert latitude & longitude into Earth-centered earth-fixed Cartesian coordinate
    z goes through 2 poles
    x cuts 0-latitude line & 0-longitude line
    y = z cross x
    
    Input:
        lat: latitude (scalar)
        long: longitude (scalar)
    
    Output:
        (x, y, z): Cartesian coordinate (list)
    """
    R = 6.3781 * 1e6  # earth radius (meter)
    z = R * sin(lat)
    x = R * cos(lat) * cos(long)
    y = R * cos(lat) * sin(long)
    
    return [x, y, z]


def _distance(p1, p2):
    """
    Calculate distance between 2 points in Cartesian space
    
    Input:
        p1 & p2: Cartesian coordiante (list)
    
    Return:
        distance between p1 & p2 (scalar)
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)
    
    
def distance_f2f(df1, df2, display=False):
    """
    Calculate distance travelled between 2 data frame
    
    Input:
        df1 & df2 (pandas.core.series.Series)
    
    Output:
        distancetravelled between df1 & df2 (scalar)
    """
    p1 = lat_long_to_XYZ(df1.lat * np.pi/180, df1.long * np.pi/180)
    p2 = lat_long_to_XYZ(df2.lat * np.pi/180, df2.long * np.pi/180)
    
    if display:
        print('p1:\t', p1)
        print('p2:\t', p2)
        
    return _distance(p1, p2)


# FUNCTION FOR CREATING TRAINING SAMPLE
def find_next_frame(now_idx, df, target_frame_dist, _threshold=5.):
    """
    Find the frame at least 0.85 * target_frame_dist meter away for frame denoted by now_idx
    
    Output:
        next_idx (int): index in DataFrame of next frame (if any)
                        -1 (if not found)
        d (float): distance between this frame and next frame
    """
    flag_found = False
    
    # initialize 
    j = now_idx + 1
    
    while not flag_found and j < len(df):
        if distance_f2f(df.iloc[j], df.iloc[j - 1]) > _threshold:  # invalid inter frame distance
            break
        
        if distance_f2f(df.iloc[j], df.iloc[now_idx]) > 0.85*target_frame_dist:
            flag_found = True
        else:
            j += 1
    
    if flag_found:
        return j, distance_f2f(df.iloc[j], df.iloc[now_idx])
    else:
        return -1, -1
    
    
def generate_sample(see_idx, df, bins_edge, data_dir, target_dist=2., f2f_dist_threshold=5., target_num_label_frame=5):
    """
    Generate sample for training set
    
    Input: 
        see_idx (int): idx of frame that is used as input
        df (pandas.DataFrame): datatframe 
        bins_edge (list): outter right edge of every bin in angle histogram
        data_dir (str): name directory contains to the image file
        target_dist (float): distance between 2 adjacent frame
        f2f_dist_threshold (float): maximum distance between 2 adjacent frame 
            (above this number -> invalide frame)
    
    Output:
        a sample as dictionary.
            'frame_name': (str) name of frame used as network input
            'travelled_dist': (list of float) distance from 1st frame to found frame whose angles used as label
            'angle_id': (list of int) 
            'angle_val': (list of float)
            
        or None if hit an invalid frame on the way
    """
    sample = {}
    sample['frame_name'] = data_dir + df.iloc[see_idx].filename
    sample['travelled_dist'] = [0.]
    sample['angle_id'] = [find_angle_id(df.iloc[see_idx].angle, bins_edge)]
    sample['angle_val'] = [df.iloc[see_idx].angle]
    
    flag_complete = False
    travelled_dist = 0
    idx = see_idx # initial value of idx
    while not flag_complete:
        # find idx of next frame
        next_idx, dist_to_next = find_next_frame(idx, df, target_dist, _threshold=f2f_dist_threshold)
        
        # check if this idx is valid
        if next_idx == -1:
            break
        
        # store travelled_dist
        travelled_dist += dist_to_next
        sample['travelled_dist'].append(travelled_dist)
        
        # store angle_value
        sample['angle_val'].append(df.iloc[next_idx].angle)
        
        # find ID of this angle
        angle_id = find_angle_id(df.iloc[next_idx].angle, bins_edge)
        # store angle_id
        sample['angle_id'].append(angle_id)
        
        # check if the sample is complete
        if len(sample['angle_id']) == target_num_label_frame:
            flag_complete = True
        else:
            # update idx
            idx = next_idx
        
    if flag_complete:
        return sample
    else:
        return None
        