#! /usr/bin/python3

from main_fit import VOC_Tool
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '2', '3'}

def train():
    while (True):
        voc = VOC_Tool('../../Train/VOC2012', ['car'], (300, 300, 3))
        voc.initModel()
        voc.loadCheckpoint('save.h5')
        voc.fit('car')

def predict(img_path):
    voc = VOC_Tool('../../Train/VOC2012', ['car'], (300, 300, 3))
    voc.loadCheckpoint('save.h5')
    voc.initModel()
    (img, ans) = voc.predict(img_path)
    voc.showPredictImg(img, ans)

def debugTrain(imgID):
    classes_list = ['car']
    voc = VOC_Tool('../../Train/VOC2012', classes_list, (300, 300, 3))
    voc.initModel(period=16, load_weight=False)
    voc.loadCheckpoint('save.h5')
    #voc.fit_single(classes_list[0], '2008_002197', epochs=256)
    voc.fit_single(classes_list[0], imgID, epochs=256)

def debugPredict(imgID):
    voc = VOC_Tool('../../Train/VOC2012', ['car'], (300, 300, 3))
    voc.loadCheckpoint('save.h5')
    voc.initModel()
    #(img, ans) = voc.predict('image/2008_002197.jpg')
    (img, ans) = voc.predict('image/' + imgID + '.jpg')
    voc.showPredictImg(img, ans)

if __name__ == "__main__":
    #train()
    #predict('image/2008_002197.jpg')
    imgID = '2008_002197'
    #debugTrain(imgID)
    #debugPredict(imgID)
