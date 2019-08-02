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
# label
# results[i][:, 0]
# conf
# results[i][:, 1]
# xmin
# results[i][:, 2]
# ymin
# results[i][:, 3]
# xmax
# results[i][:, 4]
# ymax
# results[i][:, 5]

# Input is img_resize, Output is bbox.assign_boxes(gt[key])

import main_fit
voc = main_fit.VOC_Tool('../../Train/VOC2012', ['cat', 'car'], (300, 300, 3))
voc.initModel()
voc.fit(1024, 'cat')