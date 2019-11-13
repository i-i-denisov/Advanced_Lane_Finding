import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


src = np.float32([[190, 720], [563, 470], [720, 470], [1100, 720]])
dst = np.float32([[350, 720], [350, 0], [940, 0], [940, 720]])
M = cv2.getPerspectiveTransform(src, dst)
camera_cal_pickle = pickle.load(open("mtx_dist_pickle.p", "rb"))
mtx = camera_cal_pickle["mtx"]
dist = camera_cal_pickle["dist"]
img = mpimg.imread('./extracted_images/project400.jpg')
undistorted = cv2.undistort(img, mtx, dist, None, mtx)
undistotred=cv2.line(undistorted,(190,720),(563,470),color=(255,0,0),thickness=2)
undistorted=cv2.line(undistorted,(720, 470),(1100, 720),color=(255,0,0),thickness=2)
warped=cv2.warpPerspective(undistorted,M,(img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
warped=cv2.line(warped,(350, 720),(350, 0),color=(0,255,0),thickness=2)
warped=cv2.line(warped,(940, 0),(940, 720),color=(0,255,0),thickness=2)
plt.imshow(warped)
plt.show()