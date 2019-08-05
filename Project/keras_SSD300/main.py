#! /usr/bin/python3

from main_fit import VOC_Tool

def train():
    while (True):
        voc = VOC_Tool('../../Train/VOC2012', ['car'], (300, 300, 3))
        voc.loadCheckpoint('save.h5')
        voc.initModel()
        voc.fit(4096, 'car')

def predict():
    voc = VOC_Tool('../../Train/VOC2012', ['car'], (300, 300, 3))
    voc.loadCheckpoint('save.h5')
    voc.initModel()
    voc.predict('image/test.jpg')
if __name__ == "__main__":
    #train()
    predict()