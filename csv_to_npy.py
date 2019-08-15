import sys

from training_utils import gen_classifier_dataset, gen_regressor_dataset

import numpy as np
import json
import time
import os

# python csv_to_npy.py --csv_file_name --csv_type --regressor

csv_path = sys.argv[1][2:]
csv_type = sys.argv[2][2:]
if sys.argv[3][2:] == "regressor":
    regressor = True
else:
    regressor = False

print("CSV file: ", csv_path)
print("CSV type: ", csv_type)
print("Is regressor model: ", regressor)

IMAGE_SHAPE = (200, 200, 1)
NUM_LABELS = 10
BINS_EDGE = np.load("./data/bins_edge.npy")
NUM_CLASSES = len(BINS_EDGE) - 1  

with open('./data/classes_weight.json', 'r') as fp:
    CLASSES_WEIGHT = json.load(fp)

if not regressor:
    gen_param = {'num_classes': NUM_CLASSES, 
             'num_labels': NUM_LABELS, 
             'bins_edge': BINS_EDGE, 
             'image_shape': IMAGE_SHAPE, 
             'num_samples': None, 
             'data_root_dir': "./data/training_data/", # path to folder contained images
             'flip_prob': 0.5}
    X, y = gen_classifier_dataset(csv_path, **gen_param)
    np.save('./data/CH2_%s_X.npy' % csv_type, X)
    np.save('./data/CH2_%s_y.npy' % csv_type, y)
else:
    gen_param = {'num_labels': NUM_LABELS,  
             'image_shape': IMAGE_SHAPE, 
             'num_samples': None, 
             'data_root_dir': "./data/training_data/", # path to folder contained images
             'flip_prob': 0.5}
    X, y = gen_regressor_dataset(csv_path, **gen_param)
    np.save('./data/regress_CH2_%s_X.npy' % csv_type, X)
    np.save('./data/regress_CH2_%s_y.npy' % csv_type, y)
