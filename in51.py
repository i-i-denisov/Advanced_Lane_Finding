# In[51]:
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.interpolate import splev, splrep
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False

        # poly coefs values of the last n fits of the line. First element will be zero. Do not use it
        self.recent_fits = np.array([0, 0, 0], dtype='float')

        self.good_frames = [0]

        # average x values of the fitted line over the last n iterations
        self.bestx = None

        # polynomial coefficients of last sane fit, reassigned in sanity check function
        self.best_fit = None

        # polynomial coefficients for current frame, reset at every iteration
        self.current_fit = [np.array([False])]

        # radius of curvature of the line in meters
        self.rad_curv = None

        # distance in meters of vehicle center from the line
        self.line_base_pos = None

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

        # x values for detected line pixels
        self.allx = None

        # y values for detected line pixels
        self.ally = None

        # y values for current polyfit
        self.ploty = None

        # x values for current polyfit
        self.fitx = None




#open line detection pickle
line_pickle = pickle.load(open("line_pickle.p", "rb"))
left_line = line_pickle["left_line"]
right_line = line_pickle["right_line"]
interp_len = 50
image=np.ndarray(shape=(720,1280,3),dtype=np.uint8)
image=np.zeros_like(image)


frame_grid = (left_line.good_frames[1:][-interp_len:])
#print (frame_grid, left_line.good_frames)
spline_smoothing = (interp_len + np.sqrt(2 * interp_len))
# extracting polyfit coeffs for last interp_len good fits
left_a = (left_line.recent_fits[1:, 0][-interp_len:])
right_a = (right_line.recent_fits[1:, 0][-interp_len:])
left_b = (left_line.recent_fits[1:, 1][-interp_len:])
right_b = (right_line.recent_fits[1:, 1][-interp_len:])
left_c = (left_line.recent_fits[1:, 2][-interp_len:])
right_c = (right_line.recent_fits[1:, 2][-interp_len:])
#print ((left_a), (frame_grid))
a_left_spline = splrep(frame_grid, left_a, s=spline_smoothing)
a_right_spline = splrep(frame_grid, right_a, s=spline_smoothing)
b_left_spline = splrep(frame_grid, left_b, s=spline_smoothing)
b_right_spline = splrep(frame_grid, right_b, s=spline_smoothing)
c_left_spline = splrep(frame_grid, left_c, s=spline_smoothing)
c_right_spline = splrep(frame_grid, right_c, s=spline_smoothing)
# print ((left_a),(frame_grid))

# saving smoothened coeffs for drawing
left_fit_coefs = np.vstack((splev(frame_grid, a_left_spline), splev(frame_grid, b_left_spline), splev(frame_grid, c_left_spline)))
right_fit_coefs = np.vstack(([splev(frame_grid, a_right_spline), splev(frame_grid, b_right_spline),
                   splev(frame_grid, c_right_spline)]))
plt.plot(frame_grid, left_c, 'o', frame_grid, left_fit_coefs[2], color='blue')

plt.show()
plt.plot(frame_grid, right_c, 'o', frame_grid, right_fit_coefs[2], color='red')
plt.show()

ploty=left_line.ploty
left_spline_x = np.uint16((left_fit_coefs[0,-1] * ploty ** 2 + left_fit_coefs[1,-1] * ploty + left_fit_coefs[2,-1]))
right_spline_x = np.uint16((right_fit_coefs[0,-1] * ploty ** 2 + right_fit_coefs[1,-1] * ploty + right_fit_coefs[2,-1]))

image[left_line.ally, left_line.allx] = [255, 0, 0]
image[right_line.ally, right_line.allx] = [0, 0, 255]
    # draws left and right polynomials on image
image[left_line.ploty, left_line.fitx] = [0, 0, 255]
image[right_line.ploty, right_line.fitx] = [255, 0, 0]

image[left_line.ploty, left_spline_x] = [255, 255, 255]
image[right_line.ploty, right_spline_x] = [255, 255, 255]

plt.imshow(image)
plt.show()
# plt.plot(frame_grid,left, color='blue')
# plt.plot(frame_grid,right_line.recent_fits[1:50,2], color='red')
# plt.plot(frame_grid,c_left, color='yellow')
# plt.plot(frame_grid,c_right, color='yellow')

