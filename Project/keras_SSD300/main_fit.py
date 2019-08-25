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
from ssd_fit_old import MultiboxLoss
from ssd_fit import SSDLoss
from ssd_model import SSD300
class VOC_Tool():
    '''
    Classes List not include background class
    '''
    def __init__(self, voc_path, classes_list, input_shape, 
                 checkpoint_path='./model_save/checkpoint/', save_path='./model_save/save/', priorboxPath='prior_boxes_ssd300.pkl',
                 do_crop=True, crop_area_range=[0.75, 1.0], aspect_ratio_range=[3./4, 4./3]):
        self.voc_path = voc_path
        self.checkpoint_path = checkpoint_path
        self.save_path = save_path
        self.classes_list = classes_list
        self.input_shape = input_shape
        self.classes_num = len(classes_list)
        self.classes_inf_nameprop = {}
        # <TODO>
        #self.bbox = BBoxUtility(self.classes_num + 1, pickle.load(open('prior_boxes_ssd300.pkl', 'rb')))
        priorFile = open(priorboxPath, 'rb')
        self.prior = pickle.load(priorFile)
        priorFile.close()
        self.prior = self.prior[:6537, :]
        self.bbox = BBoxUtility(self.classes_num + 1, self.prior)
        #self.bbox = BBoxUtility(self.classes_num + 1)
        # </TODO>
        self.model = SSD300(self.input_shape, self.classes_num + 1)
        self.do_crop = do_crop
        self.crop_area_range = crop_area_range
        self.aspect_ratio_range = aspect_ratio_range

        for class_name in self.classes_list:
            class_inf_nameprop = {}
            classes_list_listFilePath = voc_path + '/ImageSets/Main/' + class_name + '_train.txt'
            classes_list_listFile = open(classes_list_listFilePath)

            for nameProp in classes_list_listFile:
                nameProp_split = nameProp.split()
                class_inf_nameprop[nameProp_split[0]] = nameProp_split[1]
            
            self.classes_inf_nameprop[class_name] = class_inf_nameprop
            classes_list_listFile.close()
    
    def getOneHot(self, class_name):
        isBack = True
        oneHot = np.zeros(shape=(self.classes_num), dtype=float)
        for ptr in range(self.classes_num):
            if (self.classes_list[ptr] == class_name):
                oneHot[ptr] = 1
                break
        return oneHot
    
    def decodeOneHot(self, oneHot):
        ptr = np.argmax(oneHot)
        return self.classes_list[ptr]

    
    def getImage(self, img_ID, class_name):
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
        #img_resize = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        img = img[..., ::-1]
        #img_resize = img_resize.astype(np.float32) / 255.0
        return (img, img_height, img_width, objs_list)

    def resizeImg(self, img):
        img_resize = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        img_resize = img_resize.astype(np.float32) / 255.0
        return img_resize

    def random_sized_crop(self, img, targets):
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_area = img_w * img_h
        random_scale = np.random.random()
        random_scale *= (self.crop_area_range[1] -
                         self.crop_area_range[0])
        random_scale += self.crop_area_range[0]
        target_area = random_scale * img_area
        random_ratio = np.random.random()
        random_ratio *= (self.aspect_ratio_range[1] -
                         self.aspect_ratio_range[0])
        random_ratio += self.aspect_ratio_range[0]
        w = np.round(np.sqrt(target_area * random_ratio))     
        h = np.round(np.sqrt(target_area / random_ratio))
        if np.random.random() < 0.5:
            w, h = h, w
        w = min(w, img_w)
        w_rel = w / img_w
        w = int(w)
        h = min(h, img_h)
        h_rel = h / img_h
        h = int(h)
        x = np.random.random() * (img_w - w)
        x_rel = x / img_w
        x = int(x)
        y = np.random.random() * (img_h - h)
        y_rel = y / img_h
        y = int(y)
        img = img[y:y+h, x:x+w]
        new_targets = []
        for box in targets:
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            if (x_rel < cx < x_rel + w_rel and
                y_rel < cy < y_rel + h_rel):
                xmin = (box[0] - x_rel) / w_rel
                ymin = (box[1] - y_rel) / h_rel
                xmax = (box[2] - x_rel) / w_rel
                ymax = (box[3] - y_rel) / h_rel
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(1, xmax)
                ymax = min(1, ymax)
                box[:4] = [xmin, ymin, xmax, ymax]
                new_targets.append(box)
        new_targets = np.asarray(new_targets).reshape(-1, targets.shape[1])
        return (img, new_targets)

    def getGT(self, img_ID, class_name):
        (img, img_height, img_width, objs_list) = self.getImage(img_ID, class_name)
        gt_list = []
        gt_list_np = None
        oneHot = self.getOneHot(class_name)
        for obj in objs_list:
            gt_tmp = []
            gt_tmp.append(obj['xmin']/img_width)
            gt_tmp.append(obj['ymin']/img_height)
            gt_tmp.append(obj['xmax']/img_width)
            gt_tmp.append(obj['ymax']/img_height)
            for oh in oneHot:
                gt_tmp.append(oh)
            gt_list.append(gt_tmp)
            gt_list_np = np.array(gt_list, dtype=float)
        return gt_list_np
    
    def getAssignBoxes(self, img_ID, class_name):
        '''
        Return the True Predict Answer that the predict function should return
        '''
        gt_tmp = []
        gt = self.getGT(img_ID, class_name)
        gt = self.bbox.assign_boxes(gt)
        gt_tmp.append(gt)
        gt = np.array(gt_tmp, dtype=np.float32)
        return gt

    def getList(self, class_name):
        tmp = self.classes_inf_nameprop[class_name]
        classList = []
        for key in tmp:
            if (tmp[key] == '1'):
                classList.append(key)
        return np.array(classList)

    def getRandomChooseList(self, size, class_name):
        tmp = self.classes_inf_nameprop[class_name]
        ranList = []
        for key in tmp:
            if (tmp[key] == '1'):
                ranList.append(key)
        return np.random.choice(ranList, size=size)

    def loadCheckpoint(self, file_name, load_path = None):
        if (load_path != None):
            self.checkpoint_path = load_path
        file_path = self.checkpoint_path + file_name
        self.model.load_weights(file_path)

    def loadSave(self, file_name, load_path = None):
        if (load_path != None):
            self.save_path = load_path
        file_path = self.save_path + file_name
        self.model.load_weights(file_path)

    def initModel(self, save_freq=20):
        checkfile_name = 'save.h5'
        self.callbacks = [keras.callbacks.ModelCheckpoint(self.checkpoint_path + checkfile_name,
                                                          verbose=1,
                                                          save_weights_only=True,
                                                          #save_best_only=True,
                                                          #monitor='val_loss',
                                                          save_freq=save_freq,
                                                          load_weights_on_restart=True
                                                          )]
        #loss = SSDLoss(alpha=1.0, neg_pos_ratio=3.0)
        self.model.compile(optimizer = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                      loss = MultiboxLoss(self.classes_num, neg_pos_ratio=2.0).compute_loss,
                      #loss = loss.compute,
                      #metrics=['accuracy']
                      #metrics = loss.metrics
                      )
    
    def generator(self, class_name, batch_size=16):
        fit_list = self.getList(class_name)
        np.random.shuffle(fit_list)
        while(True):
            x = []
            y = []
            listLen = 0
            for imgID in fit_list:
                (x_tmp, tmp1, tmp2, tmp3) = self.getImage(imgID, class_name)
                gt = self.getGT(imgID, class_name)
                (x_tmp, gt) = self.random_sized_crop(x_tmp, gt)
                x_tmp = self.resizeImg(x_tmp)
                y_tmp = self.bbox.assign_boxes(gt)
                x.append(x_tmp)
                y.append(y_tmp)
                listLen += 1
                if(listLen == batch_size):
                    x = np.array(x, dtype=np.float32)
                    x = applications.keras_applications.imagenet_utils.preprocess_input(x, data_format='channels_last')
                    y = np.array(y, dtype=np.float32)
                    x_sent = x.copy()
                    y_sent = y.copy()
                    x = []
                    y = []
                    listLen = 0
                    yield (x_sent, y_sent)



    def fit(self, class_name, batch_size=8, epochs=30):
        self.initModel()
        steps_per_epoch = len(self.getList(class_name)) // batch_size
        tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.global_variables_initializer())
        self.model.fit_generator(self.generator(class_name), verbose=1, epochs=epochs,steps_per_epoch=steps_per_epoch)

    def fit_single(self, class_name, imgID, size=128, batch_size=16, epochs=10):
        '''
        # Debug Function
        This function is used to checkout if the net work
        '''
        self.initModel()
        x = []
        y = []
        (x_tmp, tmp1, tmp2, tmp3) = self.getImage(imgID, class_name)
        gt = self.getGT(imgID, class_name)
        (x_tmp, gt) = self.random_sized_crop(x_tmp, gt)
        x_tmp = self.resizeImg(x_tmp)
        y_tmp = self.bbox.assign_boxes(gt)
        for i in range(size):
            x.append(x_tmp)
            y.append(y_tmp)
        x = np.array(x, dtype=np.float32)
        x = applications.keras_applications.imagenet_utils.preprocess_input(x, data_format='channels_last')
        y = np.array(y, dtype=np.float32)

        tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.global_variables_initializer())

        self.model.fit(x, y,
                       batch_size = batch_size,
                       verbose=1,
                       epochs=epochs,
                       callbacks=self.callbacks
                       )
    
    def predict(self, img_path):
        img_list = []
        img = cv2.imread(img_path).astype(np.float32)
        img = img[..., ::-1]
        #img = self.random_sized_crop(img, None)
        img = self.resizeImg(img)
        #img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        img_list.append(img)
        img_list = np.array(img_list, dtype=np.float32)
        img_list = applications.keras_applications.imagenet_utils.preprocess_input(img_list, data_format='channels_last')

        #keras.backend.get_session().run(tf.global_variables_initializer())
        tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.global_variables_initializer())
        ans = self.model.predict(img_list)
        return (img, ans)
    
    def showPredictImg(self, img, ans):
        results = self.bbox.detection_out(ans)
        #img = image[0]
        #for i, img in enumerate(image):
        # Parse the outputs.
        det_label = results[0][:, 0]
        det_conf = results[0][:, 1]
        det_xmin = results[0][:, 2]
        det_ymin = results[0][:, 3]
        det_xmax = results[0][:, 4]
        det_ymax = results[0][:, 5]
        
        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
        
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
        
        colors = plt.cm.hsv(np.linspace(0, 1, 4)).tolist()
        
        plt.imshow(img / 255.)
        currentAxis = plt.gca()
        
        for i in range(top_conf.shape[0]):
        #for i in range(1):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            score = top_conf[i]
            label_index = int(top_label_indices[i]) - 1
            label = self.classes_list[label_index]
            display_txt = '{:0.2f}, {}'.format(score, label)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[label_index]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
        
        plt.show()