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
    voc = VOC_Tool('../../Train/VOC2012', classes_list, (300, 300, 3))
    voc.initModel()
    voc.fit_single(classes_list[0], '2007_003051', epochs=10)
    #voc.model.load_model()
    voc.model.save('save.h5')
    '''
    while (True):
        voc.loadCheckpoint('save.h5')
        voc.fit_single(classes_list[0], '2007_003051')
    '''
def debugPredict():
    voc = VOC_Tool('../../Train/VOC2012', ['car'], (300, 300, 3))
    voc.initModel()
    voc.loadCheckpoint('save.h5')
    return voc.predict('image/2007_003051.jpg')
if __name__ == "__main__":
    #train()
    #predict()
    debugTrain()