import numpy as np
import pickle
import cv2
# Filename to write
filename = "debug.txt"
# Open the file with writing permission
myfile = open(filename, 'w')

# error messages
error_poly_cant_fit = "Failed to fit a poly for line!"
error_lines_insane = "Lines failed sanity check!"

interp_len = 50  # number of frames in interpolation sequence
spline_smoothing = (interp_len - np.sqrt(2 * interp_len)) *100
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
#number of frames for averaging curvature
avg_curv_frames=25
curv=0
# loading camera calibration parameters from pickle file
camera_cal_pickle = pickle.load(open("mtx_dist_pickle.p", "rb"))
mtx = camera_cal_pickle["mtx"]
dist = camera_cal_pickle["dist"]
# calculating warp matrix
src = np.float32([[190, 720], [563, 470], [720, 470], [1100, 720]])
dst = np.float32([[350, 720], [350, 0], [940, 0], [940, 720]])
debug_text_poly = np.array([[(10, 10), (1100, 10), (1100, 60), (10, 60)]], dtype=np.int32)
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
frame_number=0
#image processing thresholds
shadow_threshold=(0,30)
l_mag_threshold=(70,200)
#sliding window config
nwindows = 9
# Choose the number of sliding windows
nwindows = 9
# Set the width of the windows +/- margin
margin = 50
# Set minimum number of pixels found to recenter window
minpix = 50

