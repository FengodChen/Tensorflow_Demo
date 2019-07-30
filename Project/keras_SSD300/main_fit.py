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

class VOC_Generator():
    def __init__(self, voc_path, classes_list):
        self.classes_num = len(classes_list)
        self.classes_list = classes_list
        self.classes_inf_name = []
        self.classes_inf_prop = []

        for ptr_class in range(self.classes_num):
            inf_name_tmp = []
            inf_prop_tmp = []
            
            classes_list_listFileName = classes_list[ptr_class]
            classes_list_listFilePath = voc_path + '/ImageSets/Main/' + classes_list_listFileName + '_train.txt'
            classes_list_listFile = open(classes_list_listFilePath)

            for nameProp in classes_list_listFile:
                nameProp_split = nameProp.split()
                inf_name_tmp.append(nameProp_split[0])
                inf_prop_tmp.append(nameProp_split[1])
            
            self.classes_inf_name.append(inf_name_tmp)
            self.classes_inf_prop.append(inf_prop_tmp)

# 初始化值
NUM_CLASSES = 1
INPUT_SHAPE = (300, 300, 3)

priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)
