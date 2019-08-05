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
import matplotlib.pyplot as plt
image = []
img = cv2.imread('image/2007_003051.jpg')
img = cv2.resize(img, (300, 300))
#img = np.array(img)
image.append(img)
voc = VOC_Tool('../../Train/VOC2012', ['car'], (300, 300, 3))
voc.loadCheckpoint('save.h5')
voc.initModel()
ans = voc.predict('image/2007_003051.jpg')
results = voc.bbox.detection_out(ans)
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
    xmin = int(round(top_xmin[i] * img.shape[1]))
    ymin = int(round(top_ymin[i] * img.shape[0]))
    xmax = int(round(top_xmax[i] * img.shape[1]))
    ymax = int(round(top_ymax[i] * img.shape[0]))
    score = top_conf[i]
    label = int(top_label_indices[i])
    display_txt = '{:0.2f}, {}'.format(score, label)
    coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
    color = colors[label]
    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
    currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})

plt.show()