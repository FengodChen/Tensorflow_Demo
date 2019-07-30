import cv2
import pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from random import shuffle

from ssd_utils import BBoxUtility
from ssd_fit import MultiboxLoss
from ssd_model import SSD300

# 初始化值
NUM_CLASSES = 1
INPUT_SHAPE = (300, 300, 3)

priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)
gt = pickle.load(open('gt_pascal.pkl', 'rb'))
keys = sorted(gt.keys())
num_train = int(round(0.8 * len(keys)))
train_keys = keys[:num_train]
val_keys = keys[num_train:]
num_val = len(val_keys)