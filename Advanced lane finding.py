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


#line object inintiation
left_line = functions.Line()
right_line = functions.Line()

fourcc = cv2.VideoWriter_fourcc(*'MP42')
cap = cv2.VideoCapture('.\input videos\project_video.mp4')
# Get frame count
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Get width and height of video stream
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('test_videos_output\project.avi', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h), True)
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
            #if (config.frame_number >= 0) & (config.frame_number < 100) & (config.frame_number != 4):
            processed_frame, left_line, right_line = functions.process_image(frame, left_line, right_line, config.interp_len)
            processed_frames += 1
            #cv2.imwrite("./extracted_images/challenge"+str(config.frame_number)+'.jpg', frame)
            out.write(processed_frame)
            # debug
            #if config.frame_number == 4 :
            #    processed_frame, left_line, right_line = functions.process_image(frame, left_line, right_line, config.interp_len, visualise=True)
            #    problem_image = frame
            #    out.write (processed_frame)
            #    processed_frames += 1
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


