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

priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))

class VOC_Generator():
    def __init__(self, voc_path, classes_list, input_shape):
        self.input_shape = input_shape
        self.classes_num = len(classes_list)
        self.classes_list = classes_list
        self.classes_inf_nameprop = {}
        self.bbox = BBoxUtility(self.classes_num, pickle.load(open('prior_boxes_ssd300.pkl', 'rb')))

        for class_name in self.classes_list:
            class_inf_nameprop = {}
            classes_list_listFilePath = voc_path + '/ImageSets/Main/' + class_name + '_train.txt'
            classes_list_listFile = open(classes_list_listFilePath)

            for nameProp in classes_list_listFile:
                nameProp_split = nameProp.split()
                class_inf_nameprop[nameProp_split[0]] = nameProp_split[1]
            
            self.classes_inf_nameprop[class_name] = class_inf_nameprop
            
#bbox_util = BBoxUtility(NUM_CLASSES, priors)
