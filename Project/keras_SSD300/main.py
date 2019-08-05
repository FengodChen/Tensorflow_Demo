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
    return voc.predict('image/test.jpg')

def debugTrain():
    classes_list = ['car', 'cat', 'dog', 'person']
    while (True):
        voc = VOC_Tool('../../Train/VOC2012', classes_list, (300, 300, 3))
        #voc.loadCheckpoint('save.h5')
        voc.initModel()
        for class_name in classes_list:
                voc.fit_single(class_name, '2007_003051')
def debugPredict():
    voc = VOC_Tool('../../Train/VOC2012', ['car'], (300, 300, 3))
    voc.loadCheckpoint('save.h5')
    voc.initModel()
    return voc.predict('image/2007_003051.jpg')
if __name__ == "__main__":
    #train()
    #predict()
    debugTrain()