import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

camera_cal_pickle = pickle.load(open("mtx_dist_pickle.p", "rb"))
mtx = camera_cal_pickle["mtx"]
dist = camera_cal_pickle["dist"]
img = mpimg.imread('./camera_cal/calibration2.jpg')
undistorted = cv2.undistort(img, mtx, dist, None, mtx)
plt.imshow(undistorted)
plt.show()