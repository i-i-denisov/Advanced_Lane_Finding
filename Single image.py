# # Single image processing

import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import config
import functions

# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.interpolate import splev, splrep
import math








# In[ ]:




# calculating warp matrix

left_line = functions.Line()
right_line = functions.Line()
frame_number = 0
# Filename to write debug info
filename = "debug.txt"
# Open the file with writing permission
myfile = open(filename, 'w')
# reading image
img = mpimg.imread('.\extracted_images\project1.jpg')

# processing image
output, left_line, right_line = functions.process_image(img, left_line, right_line,config.interp_len, True)

