import cv2
import pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#from keras.applications.keras_applications.imagenet_utils import preprocess_input
from keras import applications
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
        # <TODO>
        #self.bbox = BBoxUtility(self.classes_num + 1, pickle.load(open('prior_boxes_ssd300.pkl', 'rb')))
        prior = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
        prior = prior[:6537, :]
        #self.bbox = BBoxUtility(self.classes_num + 1, prior)
        self.bbox = BBoxUtility(self.classes_num + 1, prior)
        # </TODO>
        self.model = SSD300(self.input_shape, self.classes_num + 1)
        #self.model = SSD300(self.input_shape, self.classes_num)
        #self.bbox = None

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
        img_height = len(img)
        img_width = len(img[0])
        # TODO
        img_resize = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        return (img_resize, img_height, img_width, objs_list)
    
    def getGT(self, img_ID, class_name):
        # TODO
        # 不知道xmin...是不是就是dxmin...
        (img, img_height, img_width, objs_list) = self.getImaInf(img_ID, class_name)
        gt_list = []
        oneHot = self.getOneHot(class_name)
        for obj in objs_list:
            gt_tmp = []
            gt_tmp.append(obj['xmin']/img_width)
            gt_tmp.append(obj['ymin']/img_height)
            gt_tmp.append(obj['xmax']/img_width)
            gt_tmp.append(obj['ymax']/img_height)
            for oh in oneHot:
                gt_tmp.append(oh)
            #gt_list.append(np.array(gt_tmp, dtype=float))
            gt_list.append(gt_tmp)
            gt_list_np = np.array(gt_list, dtype=float)
        return gt_list_np
    
    def getRandomList(self, size, class_name):
        tmp = self.classes_inf_nameprop[class_name]
        ranList = []
        for key in tmp:
            if (tmp[key] == '1'):
                ranList.append(key)
        return np.random.choice(ranList, size=size)

    def initModel(self):
        self.callbacks = [keras.callbacks.ModelCheckpoint('./model_save/checkpoint/save.h5',
                                             verbose=1,
                                             save_weights_only=True)]
        self.model.compile(optimizer = keras.optimizers.Adam(3e-4),
                      loss = MultiboxLoss(self.classes_num, neg_pos_ratio=2.0).compute_loss
                      #metrics=['accuracy']
                      )
    def fit(self, size, class_name, batch_size=8, epochs=10):
        self.initModel()
        fit_list = self.getRandomList(size, class_name)
        x = []
        y = []
        for imgID in fit_list:
            (x_tmp, tmp1, tmp2, tmp3) = self.getImaInf(imgID, class_name)
            x.append(x_tmp)
            y_tmp = self.bbox.assign_boxes(self.getGT(imgID, class_name))
            y.append(y_tmp)
        x = np.array(x, dtype=np.float32)
        x = applications.keras_applications.imagenet_utils.preprocess_input(x, data_format='channels_last')
        y = np.array(y, dtype=np.float32)

        keras.backend.get_session().run(tf.global_variables_initializer())

        self.model.fit(x, y,
                       batch_size = batch_size,
                       verbose=1,
                       epochs=epochs,
                       callbacks=self.callbacks)

    
    
    '''
    def setPriors(self):
        #TODO
        pass

    def setBBox(self):
        #TODO
        pass
    '''