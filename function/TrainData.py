from random import random

def getRandomList(length, multi):
    rList = []
    for i in range(length):
        rList.append(random()*multi)
    return rList

def getTrainData(length, w_true):
    data_x = getRandomList(length, 100)
    data_y = 0
    for i in range(length):
        data_y += w_true[i] * data_x[i]
    return [data_x, data_y]
