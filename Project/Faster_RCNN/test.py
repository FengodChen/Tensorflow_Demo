import os
import tensorflow as tf
import numpy as np
import visualize
import cv2
img = cv2.imread('image_demo_random.png')
import matplotlib.image as mpimg
image = mpimg.imread("image_demo_random.png")
bbox = np.array([[10,10,20,20]])
class_ids = [0]
class_names = ['cat']
visualize.display_instances(image, bbox, class_ids, class_names)
visualize.draw_boxes(image, bbox)