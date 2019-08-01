import ssd_utils
import pickle

p = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
gt = pickle.load(open('gt_pascal.pkl', 'rb'))
y = gt['frame04467.png']
bb = ssd_utils.BBoxUtility(4, p)
yy = bb.assign_boxes(y)