import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
def camera_calibrate(cal_images_path='./camera_cal/calibration*.jpg', nx=9, ny=6):
    # define constant values
    # number of inside corners on calibration chessboard pattern
    nx = 9
    ny = 6
    # pickle filename to save calibration result matrices
    cal_pickle = 'mtx_dist_pickle.p'
    # read a list of calibration images
    cal_images = glob.glob(cal_images_path)

    # arrays for imagepoints and object points
    objpoints = []  # 3d points in real space
    imgpoints = []  # 2d points on image
    # defining object used for calibration
    objp = np.zeros(((nx) * (ny), 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,
                                                 2)  # fill x,y coords with values 0,0; 1,0; ...; 6,5;7,5 corresponding to chessboard corners, z-coord is zero as our object used for calibration is flat chessboard
    # number of successfuly processed calibration images
    good_cal_images = 0
    # loop through calibration images
    for fname in cal_images:
        # read an image from file
        img = mpimg.imread(fname)
        # convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # plt.imshow (gray,cmap='gray')
        # find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # if corners sucessfully found then add them to calibration arrays
        if ret:
            imgpoints.append(corners)
            objpoints.append(
                objp)  # in this case all objects are same chessboard pattern, but we can use different ones or some 3d object from differennt perrspectives
            good_cal_images += 1

    print (good_cal_images, 'from', len(cal_images), 'images are good for calibration')

    # calculating disrortion correction matrices. Note that it uses img.shape[1::-1] value as image size that means we take dimensions of last image from calibration collection assuming as all of them are taken with same camera and have same resolution
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)

    if ret:
        output = open(cal_pickle, 'wb')
        pickle.dump({"mtx": mtx, "dist": dist}, output, 2)
        print ("Calibration paramaters saved to pickle", cal_pickle)
        output.close()
    return ret, mtx, dist


# In[ ]:


ret, mtx, dist = camera_calibrate()
if ret:
    print ('Camera calibration successful')


# In[10]:


camera_cal_pickle = pickle.load(open("mtx_dist_pickle.p", "rb"))
mtx = camera_cal_pickle["mtx"]
dist = camera_cal_pickle["dist"]
# calculating warp matrix
src = np.float32([[190, 720], [563, 470], [720, 470], [1100, 720]])
dst = np.float32([[350, 720], [350, 0], [940, 0], [940, 720]])
debug_text_poly = np.array([[(10, 10), (1100, 10), (1100, 60), (10, 60)]], dtype=np.int32)
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
img = mpimg.imread('./extracted_images/project554.jpg')

kernel_size = 7  # Must be an odd number (3, 5, 7...)

# reading in an image
imshape = img.shape
# undistorting image using saved matrices
undistorted = cv2.undistort(img, mtx, dist, None, mtx)

warped = cv2.warpPerspective(undistorted, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
# drawing calibation lines on undistorted image
lines = np.copy(undistorted)
cv2.line(lines, (190, 720), (555, 470), color=[0, 0, 255], thickness=2)
cv2.line(lines, (1117, 720), (720, 470), color=[0, 0, 255], thickness=2)

warped = cv2.warpPerspective(undistorted, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)

warped_lines = np.copy(warped)
cv2.line(warped_lines, (380, 720), (380, 0), color=[0, 0, 255], thickness=2)
cv2.line(warped_lines, (950, 720), (950, 0), color=[0, 0, 255], thickness=2)

detected_lines, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
detected_lines.tight_layout()
ax1.imshow(lines)
ax1.set_title('color_warped', fontsize=50)
ax2.imshow(warped_lines, cmap="inferno")
ax2.set_title('warped_lines_drawn', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

