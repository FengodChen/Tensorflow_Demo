import cv2
import pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from random import shuffle
from xml.dom import minidom

from ssd_utils import BBoxUtility
from ssd_fit import MultiboxLoss
from ssd_model import SSD300

class VOC_Generator():
    def __init__(self, voc_path, classes_list, input_shape):
        self.voc_path = voc_path
        self.classes_list = classes_list
        self.input_shape = input_shape
        self.classes_num = len(classes_list)
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
    
    def getImaAndGT(self, img_ID, obj_Name):
        objs_list = []
        img_path = self.voc_path + '/JPEGImages/' + img_ID + '.jpg'
        img_xml = self.voc_path + '/Annotations/' + img_ID + '.xml'
        img = cv2.imread(img_path)
        inf = minidom.parse(img_xml).documentElement
        objs = inf.getElementsByTagName('object')
        for obj in objs:
            if (obj.getElementsByTagName('name')[0].childNodes[0].data == obj_Name):
                obj_inf = {}
                obj_inf['xmax'] = int(obj.getElementsByTagName('xmax')[0].childNodes[0].data)
                obj_inf['xmin'] = int(obj.getElementsByTagName('xmin')[0].childNodes[0].data)
                obj_inf['ymax'] = int(obj.getElementsByTagName('ymax')[0].childNodes[0].data)
                obj_inf['ymin'] = int(obj.getElementsByTagName('ymin')[0].childNodes[0].data)
                objs_list.append(obj_inf)
        return (img, objs_list)