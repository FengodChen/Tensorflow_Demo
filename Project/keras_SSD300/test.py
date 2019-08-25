'''
import ssd_utils
import pickle
import numpy as np


# p.shape = (7308, 8)
# p[i] = [xmin, ymin, xmax, ymax, varxc, varyc, varw, varh].
# Always (varxc, varyc, varw, varh) = (0.1, 0.1, 0.2, 0.2)
p = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
# gt[key].shape = (x, 7)
# x means box number
# 7 means classes_number + 4, classes number not include background number
# (+4) means dxmin, dymin, dxmax, dymax
# So gt = [dxmin, dymin, dxmax, dymax, (classes one-hot array)]
gt = pickle.load(open('gt_pascal.pkl', 'rb'))
y = gt['frame04467.png'].copy()
# Class number = 4, include background class
bb = ssd_utils.BBoxUtility(4, p)
# yy.shape = (7308, 16)
yy = bb.assign_boxes(y)

targets = []
targets.append(yy)
tmp_targets = np.array(targets)
results = bb.detection_out(tmp_targets)

# Input is img_resize, Output is bbox.assign_boxes(gt[key])
'''

'''
import main_fit
#voc = main_fit.VOC_Tool('../../Train/VOC2012', ['cat', 'car', 'dog', 'person'], (300, 300, 3))
voc = main_fit.VOC_Tool('../../Train/VOC2012', ['car'], (300, 300, 3))
#voc.loadCheckpoint('save.h5')
voc.initModel()
voc.fit(1024, 'car')
'''

from main_fit import VOC_Tool
import cv2
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from keras.preprocessing import image as imp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '2', '3'}
image = []

x_tmp = cv2.imread('image/2008_002197.jpg')
#x_tmp = cv2.resize(x_tmp, (300, 300))
x_tmp = x_tmp[...,::-1]
x_tmp = x_tmp.astype(np.float32)

img = imp.load_img('image/2008_002197.jpg', (300, 300))
img = imp.img_to_array(img)
image.append(img)

voc = VOC_Tool('../../Train/VOC2012', ['car', 'person'], (300, 300, 3))
voc.loadCheckpoint('save.h5')
voc.initModel()
(img, ans) = voc.predict('image/2008_002197.jpg')
gt = voc.getAssignBoxes('2008_002197', 'car')
# <Debug>
# This is not very influential for the answer
#(x_tmp, gt) = voc.random_sized_crop(x_tmp, gt)
# </Debug>

# <TODO id=190820001>
# <Debug>
'''
for ptr in range(len(voc.prior)):
    print('prior[{}]:{}'.format(ptr, voc.prior[ptr, :]))
    print('pre[{}]:{}'.format(ptr, ans[0][ptr, -8:]))
    input('(ENTER)')


ptr = 0
for i in gt[0,:,5]:
    if (i > 0.85):
        print('gt[{}]:{}'.format(ptr, gt[0][ptr]))
        print('pre[{}]:{}'.format(ptr, ans[0][ptr]))
    ptr += 1

ptr = 0
for i in ans[0,:,5]:
    if (i > 0.85):
        print('pre[{}]:{}'.format(ptr, i))
        print('gt[{}]:{}'.format(ptr, gt[0,ptr,5]))
    ptr += 1
'''
# </Debug>
# </TODO>

#voc.showPredictImg(image[0], gt)
voc.showPredictImg(img, ans)
