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

class VOC_Tool():
    '''
    Classes List not include background class
    '''
    def __init__(self, voc_path, classes_list, input_shape):
        self.voc_path = voc_path
        self.classes_list = classes_list
        self.input_shape = input_shape
        self.classes_num = len(classes_list)
        self.classes_inf_nameprop = {}
        #self.bbox = BBoxUtility(self.classes_num, pickle.load(open('prior_boxes_ssd300.pkl', 'rb')))
        self.bbox = None

        for class_name in self.classes_list:
            class_inf_nameprop = {}
            classes_list_listFilePath = voc_path + '/ImageSets/Main/' + class_name + '_train.txt'
            classes_list_listFile = open(classes_list_listFilePath)

            for nameProp in classes_list_listFile:
                nameProp_split = nameProp.split()
                class_inf_nameprop[nameProp_split[0]] = nameProp_split[1]
            
            self.classes_inf_nameprop[class_name] = class_inf_nameprop
    
    def getOneHot(self, class_name):
        oneHot = np.zeros(shape=(self.classes_num), dtype=float)
        for ptr in range(self.classes_num):
            if (self.classes_list[ptr] == class_name):
                oneHot[ptr] = 1
                break
        return oneHot
    
    def decodeOneHot(self, oneHot):
        ptr = np.argmax(oneHot)
        return self.classes_list[ptr]

    
    def getImaInf(self, img_ID, class_name):
        objs_list = []
        img_path = self.voc_path + '/JPEGImages/' + img_ID + '.jpg'
        img_xml = self.voc_path + '/Annotations/' + img_ID + '.xml'
        img = cv2.imread(img_path)
        inf = minidom.parse(img_xml).documentElement
        objs = inf.getElementsByTagName('object')
        for obj in objs:
            if (obj.getElementsByTagName('name')[0].childNodes[0].data == class_name):
                obj_inf = {}
                obj_inf['xmax'] = int(obj.getElementsByTagName('xmax')[0].childNodes[0].data)
                obj_inf['xmin'] = int(obj.getElementsByTagName('xmin')[0].childNodes[0].data)
                obj_inf['ymax'] = int(obj.getElementsByTagName('ymax')[0].childNodes[0].data)
                obj_inf['ymin'] = int(obj.getElementsByTagName('ymin')[0].childNodes[0].data)
                objs_list.append(obj_inf)
        return (img, objs_list)
    
    def getGT(self, img_ID, class_name):
        # TODO
        # 不知道xmin...是不是就是dxmin...
        (img, objs_list) = getImaInf(img_ID, class_name)
        gt_list = []
        oneHot = getOneHot(class_name)
        img_height = len(img)
        img_width = len(img[0])
        for obj in objs_list:
            gt_tmp = []
            gt_tmp.append(obj['xmin']/img_width)
            gt_tmp.append(obj['ymin']/img_height)
            gt_tmp.append(obj['xmax']/img_width)
            gt_tmp.append(obj['ymax']/img_height)
            for oh in oneHot:
                gt_tmp.append(oh)
            gt_list.append(np.array(gt_tmp, dtype=float))
        return gt_list
    
    
    def getPriors(self):
        #TODO
        pass

    def setBBox(self):
        #TODO
        pass