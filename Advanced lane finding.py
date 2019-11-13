#!/usr/bin/env python
# coding: utf-8

# # Import Packages

# In[1]:


# importing some useful packages

import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

import functions






import config
#interp_len = 50  # number of frames in interpolation sequence
#spline_smoothing = (interp_len - np.sqrt(2 * interp_len)) / 10
#ym_per_pix = 30 / 720  # meters per pixel in y dimension
#xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
# loading camera calibration parameters from pickle file
#camera_cal_pickle = pickle.load(open("mtx_dist_pickle.p", "rb"))
#mtx = camera_cal_pickle["mtx"]
#dist = camera_cal_pickle["dist"]
# calculating warp matrix
#src = np.float32([[184, 720], [592, 450], [688, 450], [1117, 720]])
#dst = np.float32([[350, 720], [350, 0], [910, 0], [910, 720]])
#debug_text_poly = np.array([[(10, 10), (1100, 10), (1100, 60), (10, 60)]], dtype=np.int32)
#M = cv2.getPerspectiveTransform(src, dst)
#Minv = cv2.getPerspectiveTransform(dst, src)


# Filename to write
#filename = "debug.txt"
# Open the file with writing permission
#config.myfile = open(filename, 'w')

#line object inintiation
left_line = functions.Line()
right_line = functions.Line()

fourcc = cv2.VideoWriter_fourcc(*'MP42')
cap = cv2.VideoCapture('.\challenge_video.mp4')
# Get frame count
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Get width and height of video stream
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('test_videos_output\challenge.avi', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h), True)
config.frame_number = 0
if out.isOpened():

    processed_frames = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        else:
            #if config.frame_number > 600:
             # break
            # problem_image=frame
            if (config.frame_number >= 0) & (config.frame_number < 100) & (config.frame_number != 4):
                processed_frame, left_line, right_line = functions.process_image(frame, left_line, right_line, config.interp_len)
                processed_frames += 1
                cv2.imwrite("./extracted_images/challenge"+str(config.frame_number)+'.jpg', frame)
                out.write(processed_frame)
            # debug
            if config.frame_number == 4 :
                processed_frame, left_line, right_line = functions.process_image(frame, left_line, right_line, config.interp_len, visualise=True)
                problem_image = frame
                out.write (processed_frame)
                processed_frames += 1
            config.frame_number += 1
cap.release()
out.release()
print('processsed ', config.frame_number, "frames")
# Close the file
config.myfile.close()

#saving line detection parameters
line_pickle = "line_pickle.p"
output = open(line_pickle, 'wb')
pickle.dump({"left_line": left_line, "right_line": right_line}, output, 2)
output.close()

# print (left_line.recent_fits)
# print (right_line.recent_fits)
# drawing original image and processed image
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(frame)
# ax1.set_title('Original image', fontsize=50)
# ax2.imshow(processed_frame)
# ax2.set_title('Undistorted and Warped Image', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
