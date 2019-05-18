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
    
    
def find_angle_id(idx, bin_edges, df, id_column="angle_id"):
    """
    Find id for idx-th row of center_cam_df w.r.t this row's steering angle
    
    Input:
        idx (scalar): row index of center_cam_df 
        df (pd.Dataframe): dataframe  
        bin_edges (np.array): array of bin edges 
        id_column (str): name of column of df which stores steering angle ID
    
    Output:
        assign steering class to inputted row
    """
    angle = df.iloc[idx].angle
    i = 0
    flag_found_bin = False
    while i < len(bin_edges) - 1 and not flag_found_bin:
        if angle >= bin_edges[i] and angle < bin_edges[i + 1]:
            flag_found_bin = True
            df[id_column].iat[idx] = i  # use iat[] to make sure the change is made to center_cam_df 
            # itself, not a copy of center_cam_df
        i += 1
    
    if not flag_found_bin:
        # not found any bin contains this steering angle --> equal to the right edge of the last bin
        df[id_column].iat[idx] = i - 1
    
    
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
def find_next_frame(now_idx, df, _threshold=5.):
    """
    Find the frame at least 0.85 meter away for frame denoted by now_idx
    
    Output:
        next_idx (int): index in DataFrame of next frame (if any)
                        -1 (if not found)
        d (float): distance between this frame and next frame
    """
    flag_found = False
    # initialize 
    j = now_idx + 1
    d = 0  
    while not flag_found and j < len(df):
        d = distance_f2f(df.iloc[j], df.iloc[now_idx])
        if d > _threshold:  # invalid inter frame distance
            break
        if distance_f2f(df.iloc[j], df.iloc[now_idx]) > 0.85:
            flag_found = True
        else:
            j += 1
    
    if flag_found:
        return j, d
    else:
        return -1, -1


def generate_training_sample(now_idx, df, path_prefix, len_spatial_history=10):
    """
    Generate one training sample
    """    
    
    # initialize frame_list, angle_id & angle_list
    frame_list = [path_prefix + df["filename"].iloc[now_idx]]
    angle_list = [df["angle"].iloc[now_idx]]
    angle_id_list = [df["angle_id"].iloc[now_idx]]
    travelled_dist_list = [0]
    
    flag_incomplete_sample = False
    
    for i in range(len_spatial_history):
        next_idx, d = find_next_frame(now_idx, df)
        if next_idx > 0:  # valid next_idx
            # update now_idx with next_idx
            now_idx = next_idx
            
            # record file name & angle
            frame_list.append(path_prefix + df["filename"].iloc[now_idx])
            angle_list.append(df["angle"].iloc[now_idx])
            angle_id_list.append(df["angle_id"].iloc[now_idx])
            travelled_dist_list.append(travelled_dist_list[-1] + d)

        else:
            # next_idx is a filtered out index --> can't find next frame
            flag_incomplete_sample = True
            break
    
    if flag_incomplete_sample:
#         print("Make an incomplete sample. Return None")
        return None
    else:
        out = {'frame_list': frame_list,
               'angle_list': angle_list, 
               'angle_id_list': angle_id_list,
               'travelled_dist_list': travelled_dist_list}
        return out
 
