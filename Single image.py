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

#plot, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#plot.tight_layout()
#ax1.imshow(img, cmap='inferno')
##ax1.set_title('image', fontsize=50)
#ax2.imshow(output, cmap="inferno")
#ax2.set_title('processed image', fontsize=50)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#print (left_line.rad_curv, right_line.rad_curv)
#print (left_line.best_fit, right_line.best_fit)
#plt.show()

#src=np.array([[[190, 720], [563, 470], [720, 470], [1100, 720]]])
#test=np.zeros_like(img)
#test=cv2.fillPoly(test,src,(0, 255, 0))
#plt.imshow (output)
#plt.show()
#dst=np.array([[[350, 720], [350, 0], [940, 0], [940, 720]]])
#test=np.zeros_like(img)
#test=cv2.fillPoly(test,dst,(0, 255, 0))
#newwarp = cv2.warpPerspective(test, Minv, (img.shape[1], img.shape[0]),flags=cv2.INTER_LINEAR)
#plt.imshow (newwarp)
#plt.show()