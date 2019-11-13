
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import config
from scipy.interpolate import splev, splrep

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False

        # poly coefs values of the last n fits of the line. First element will be zero. Do not use it
        self.recent_fits = np.array([0, 0, 0], dtype='float')
        #array of frames with succcessfull lane detection
        self.good_frames = [0]

        # average x values of the fitted line over the last n iterations
        #self.bestx = None

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

# Functions definition

def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    # keeping in mind that we will pass L-channel or grayscale image
    # gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray = np.copy(image)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if (orient != 'x') and (orient != 'y'):
        print('Error in function abs_sobel_thresh, orientation should be "x" or "y" value')
        return None
    derivative_direction = dict(x=(1, 0), y=(0, 1))
    # derivative_direction
    # ['x']=(1,0)
    # ['y']=(0,1)
    # print (derivative_direction[orient])
    sobel = cv2.Sobel(gray, cv2.CV_64F, *derivative_direction[orient], ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    return sobel_binary
    # return scaled_sobel

def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    # keeping in mind that we will pass L-channel or grayscale image
    # gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray = np.copy(image)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
    # 3) Calculate the magnitude
    sobel_mag = np.sqrt(sobelx * sobelx + sobely * sobely)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel_mag = np.uint8(255 * sobel_mag / np.max(sobel_mag))
    # 5) Create a binary mask where mag thresholds are met
    sobel_binary_mag = np.zeros_like(scaled_sobel_mag)
    sobel_binary_mag[(scaled_sobel_mag >= thresh_min) & (scaled_sobel_mag <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    return sobel_binary_mag
    # return scaled_sobel_mag

def layer_threshold(image, kernel_size=3, thresh=(0, 255)):
    blur_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    s_binary = np.zeros_like(image)
    s_binary[(blur_image >= thresh[0]) & (blur_image <= thresh[1])] = 1
    return s_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Grayscale
    # keeping in mind that we will pass L-channel or grayscale image
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = np.copy(image)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    sobel_binary_dir = np.uint8(np.zeros_like(sobelx))
    sobel_binary_dir[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # Return the bina
    return sobel_binary_dir
    # graphics_grad_dir=np.uint8(absgraddir/(np.pi/2)*255)
    # return graphics_grad_dir

def sliding_window(binary_warped, right_line, left_line):
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    config.nwindows = 9
    # Set the width of the windows +/- margin
    config.margin = 100
    # Set minimum number of pixels found to recenter window
    config.minpix = 50

    # creating line objects

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // config.nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(config.nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - config.margin
        win_xleft_high = leftx_current + config.margin
        win_xright_low = rightx_current - config.margin
        win_xright_high = rightx_current + config.margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        # print (type(good_left_inds))
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        ### If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > config.minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > config.minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    left_line.allx = nonzerox[left_lane_inds]
    left_line.ally = nonzeroy[left_lane_inds]
    right_line.allx = nonzerox[right_lane_inds]
    right_line.ally = nonzeroy[right_lane_inds]

    return out_img, left_line, right_line

def find_lines(binary_warped, left_line, right_line):


    # reset pixels for left and right line
    img_shape = binary_warped.shape
    left_line.ploty = np.mgrid[0:img_shape[0]]
    right_line.ploty = np.mgrid[0:img_shape[0]]
    left_line.fitx = None
    right_line.fitx = None

    # extracting data from left_line and right_line objects, maybe should be revised
    # TODO check how it handles empy best fit arrays
    left_line.current_fit = [np.array([False])]
    right_line.current_fit = [np.array([False])]

    # Find our lane pixels first
    # if we know lane positions from previous frame
    if (left_line.detected & right_line.detected):
        # reset detection flags
        left_line.detected = False
        right_line.detected = False
        # searching for line pixels for new frame
        config.myfile.write((str(config.frame_number) + " doing around poly fit\n"))
        out_img, left_line, right_line = search_around_poly(binary_warped, left_line, right_line)
        # reassignment of line poly fit coeffs
        left_line, right_line = fit_poly(binary_warped.shape, left_line, right_line)
        # perform sanity check
        left_line, right_line = sanity_check(left_line, right_line)
        # if detected lines are insane then handle error and perform sliding window
        if not (left_line.detected & right_line.detected):
            config.myfile.write((str(config.frame_number) + ' around poly sanity check failed,doing sliding window\n'))
            out_img, left_line, right_line = sliding_window(binary_warped, left_line, right_line)
            left_line, right_line = fit_poly(binary_warped.shape, left_line, right_line)
    else:
        # reset detection flags
        left_line.detected = False
        right_line.detected = False
        # sliding window line search
        config.myfile.write((str(config.frame_number) + " doing sliding window\n"))
        out_img, left_line, right_line = sliding_window(binary_warped, left_line, right_line)
        left_line, right_line = fit_poly(binary_warped.shape, left_line, right_line)

    # TODO rewrite after sanity check is implemented
    left_line, right_line = sanity_check(left_line, right_line)

    ## Visualization ##
    # Colors in the left and right lane regions

    out_img[left_line.ally, left_line.allx] = [255, 0, 0]
    out_img[right_line.ally, right_line.allx] = [0, 0, 255]
    # draws left and right polynomials on image
    out_img[left_line.ploty, left_line.fitx] = [0, 0, 255]
    out_img[right_line.ploty, right_line.fitx] = [255, 0, 0]

    # if lines passed sanity check
    if (left_line.detected & right_line.detected):
        # adding fit coeffs to historical array
        left_line.recent_fits = np.vstack((left_line.recent_fits, left_line.current_fit))
        right_line.recent_fits = np.vstack((right_line.recent_fits, right_line.current_fit))
        left_line.good_frames = np.hstack((left_line.good_frames, config.frame_number))
        right_line.good_frames = np.hstack((right_line.good_frames, config.frame_number,))


        # calculating curvature and offset
        # definig curvature evaluation point as bottom of the image
        y_eval = (binary_warped.shape[0] - 1)
        left_line.rad_curv = radius_of_curv(left_line.best_fit,y_eval)
        right_line.rad_curv = radius_of_curv(right_line.best_fit,y_eval)
        #calculating car offset
        left_line.line_base_pos = (left_line.fitx[y_eval] - binary_warped.shape[1] // 2) * config.xm_per_pix
        right_line.line_base_pos = (right_line.fitx[y_eval] - binary_warped.shape[1] // 2) * config.xm_per_pix
    else:
        print(config.frame_number, "line detection failed")

    # print (out_img.shape,type(out_img))
    return out_img, left_line, right_line

def fit_poly(img_shape, left_line, right_line):
    leftx = left_line.allx
    lefty = left_line.ally
    rightx = right_line.allx
    righty = right_line.ally
    # fitting second order polynomial to left and right line pixel sets
    try:
        left_line.current_fit = np.polyfit(lefty, leftx, 2)
        right_line.current_fit = np.polyfit(righty, rightx, 2)
        config.myfile.write((str(config.frame_number) + "polyfit successful\n"))
    except TypeError:
        config.myfile.write((str(config.frame_number) + "polyfit unsuccessful\n"))
        left_line.detected = False
        right_line.detected = False
        left_line.current_fit = [np.array([False])]
        right_line.current_fit = [np.array([False])]
        return left_line, right_line

    # I am not sure if I need this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ###visualisation
    # Generate x and y values for plotting
    left_fit_coefs = left_line.current_fit
    right_fit_coefs = right_line.current_fit
    # print (left_fit_coefs,right_fit_coefs)

    try:
        left_line.fitx = np.uint16(
            (left_fit_coefs[0] * left_line.ploty ** 2 + left_fit_coefs[1] * left_line.ploty + left_fit_coefs[2]))
        right_line.fitx = np.uint16(
            (right_fit_coefs[0] * right_line.ploty ** 2 + right_fit_coefs[1] * right_line.ploty + right_fit_coefs[2]))
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        config.myfile.write((str(config.frame_number) + config.error_poly_cant_fit + "\n"))
        #    #reset detection flag if np.polyfit couldn't fit lines
        left_line.detected = False
        right_line.detected = False
        left_line.current_fit = [np.array([False])]
        right_line.current_fit = [np.array([False])]
        return left_line, right_line
    #    left_fitx = np.uint16(1*left_line.ploty**2 + 1*left_line.ploty)
    #    right_fitx = np.uint16(1*right_line.ploty**2 + 1*right_line.ploty)
    #    #cv2.putText(out_img, config.error_poly_cant_fit ,(10,10), font, 1,(0,0,255),2,cv2.LINE_AA)
    # checking poly fit calculations fit to image boundaries
    left_line.fitx[left_line.fitx > (img_shape[1] - 1)] = img_shape[1] - 1
    right_line.fitx[right_line.fitx > (img_shape[1] - 1)] = img_shape[1] - 1
    left_line.fitx[left_line.fitx < 0] = 0
    right_line.fitx[right_line.fitx < 0] = 0
    return left_line, right_line  # , ploty,left_fit_coefs,right_fit_coefs

def search_around_poly(binary_warped, left_line, right_line):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    left_fit_coefs = left_line.best_fit
    right_fit_coefs = right_line.best_fit

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    left_lane_inds = (nonzerox >= left_fit_coefs[0] * (nonzeroy ** 2) + left_fit_coefs[1] * nonzeroy + left_fit_coefs[
        2] - margin) & (nonzerox <= left_fit_coefs[0] * (nonzeroy ** 2) + left_fit_coefs[1] * nonzeroy + left_fit_coefs[
        2] + margin)
    right_lane_inds = (nonzerox >= right_fit_coefs[0] * (nonzeroy ** 2) + right_fit_coefs[1] * nonzeroy +
                       right_fit_coefs[2] - margin) & (
                              nonzerox <= right_fit_coefs[0] * (nonzeroy ** 2) + right_fit_coefs[1] * nonzeroy +
                              right_fit_coefs[2] + margin)
    # print (left_lane_inds)
    # Again, extract left and right line pixel positions
    left_line.allx = nonzerox[left_lane_inds]
    left_line.ally = nonzeroy[left_lane_inds]
    right_line.allx = nonzerox[right_lane_inds]
    right_line.ally = nonzeroy[right_lane_inds]
    # leftx = nonzerox[left_lane_inds]
    # lefty = nonzeroy[left_lane_inds]
    # rightx = nonzerox[right_lane_inds]
    # righty = nonzeroy[right_lane_inds]

    # to delete
    # Fit new polynomials
    # try:
    #    left_fitx, right_fitx, ploty,left_fit_coefs,right_fit_coefs = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    # except TypeError:
    # leftx, lefty not defined or poly could not fit
    #    print("leftx, lefty not defined or poly could not fit")

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    ploty = np.mgrid[0:binary_warped.shape[0]]
    left_fitx = left_fit_coefs[0] * ploty ** 2 + left_fit_coefs[1] * ploty + left_fit_coefs[2]
    right_fitx = right_fit_coefs[0] * ploty ** 2 + right_fit_coefs[1] * ploty + right_fit_coefs[2]
    # print (ploty)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    ## End visualization steps ##

    return result, left_line, right_line

def sanity_check(left_line, right_line):
    ### TODO: Add line sanity check they are close to parallel, distance between is in order of lane width, similar curvature
    # print (left_line.current_fit,right_line.current_fit)
    if (len(left_line.current_fit) == 3) & (len(right_line.current_fit) == 3):
        left_line.detected = True
        right_line.detected = True

        max_dist = np.amax(right_line.fitx - left_line.fitx)
        min_dist = np.amin(right_line.fitx - left_line.fitx)
        a = left_line.current_fit[0] / right_line.current_fit[0]
        b = left_line.current_fit[1] / right_line.current_fit[1]
        if (max_dist > 750) | (min_dist < 370):
            left_line.detected = False
            right_line.detected = False
            print("Sanity check failed, Frame ", config.frame_number, ". Distance between lines too big\small. Max Distance ",
                  max_dist, " Min distance ", min_dist)
            return left_line, right_line
        if ((a > 10) | (a < 0.1) | (b > 10) | (b < 0.1)) & (max_dist > 650) & (min_dist < 550):
            left_line.detected = False
            right_line.detected = False
            print(
                "Sanity check failed, Frame ", config.frame_number, ". Polyfit not parallel. Left_line ",
                left_line.current_fit,
                " Right_line ", right_line.current_fit)
            print(max_dist, min_dist)
            # if all sanity checks passed then reassign best fit
        if (max_dist < 600) & (min_dist > 500):
            left_line.detected = True
            right_line.detected = True
    # if all sanity checks passed then reassign best fit
    if left_line.detected & right_line.detected:
        left_line.best_fit = left_line.current_fit
        right_line.best_fit = right_line.current_fit

    return left_line, right_line

def avg_curv(left_line,right_line,frames_number=25):
    left_last_fits=left_line.recent_fits[1:][-frames_number:]
    right_last_fits = right_line.recent_fits[1:][-frames_number:]
    left_avg_fit=np.average(left_last_fits,axis=0)
    right_avg_fit=np.average(right_last_fits,axis=0)
    left_avg_curv=radius_of_curv(left_avg_fit,left_line.ploty[-1])
    right_avg_curv = radius_of_curv(right_avg_fit, right_line.ploty[-1])
    avg=(left_avg_curv+right_avg_curv)/2
    return avg

def radius_of_curv(fit_coefs,y_eval):
    try:
        scld_fit = [fit_coefs[0] * config.xm_per_pix / (config.ym_per_pix ** 2),fit_coefs[1] * config.xm_per_pix,fit_coefs[2] * config.xm_per_pix]

        rad_curv = ((1 + (2 * scld_fit[0] * y_eval * config.ym_per_pix + scld_fit[1]) ** 2) ** 1.5) / (2 * np.abs(scld_fit[0]))
    except:
        rad_curv=0
    return rad_curv


def process_image(image, left_line, right_line, interp_len, visualise=False):
    # gaussian blur kernel size
    kernel_size = 7  # Must be an odd number (3, 5, 7...)

    # reading in an image
    imshape = image.shape
    # undistorting image using saved matrices
    undistorted = cv2.undistort(image, config.mtx, config.dist, None, config.mtx)

    # converting image to HSL space
    HLS = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HLS)
    # extracting s chanel
    s_channel = HLS[:, :, 2]
    l_channel = HLS[:, :, 1]
    # blur_HSL = cv2.GaussianBlur(HSL,(kernel_size, kernel_size),0)

    # calculating defferent thresholds to L-channel of image
    l_gradx = abs_sobel_thresh(l_channel, orient='x', sobel_kernel=kernel_size, thresh=(40, 150))
    l_grady = abs_sobel_thresh(l_channel, orient='y', sobel_kernel=kernel_size, thresh=(0, 40))
    l_mag_binary = mag_thresh(l_channel, sobel_kernel=9, thresh=config.l_mag_threshold)
    l_dir_binary = dir_threshold(l_channel, sobel_kernel=15, thresh=(0, 1.3))
    # get simple s-layer threshold
    s_binary = layer_threshold(s_channel, kernel_size=kernel_size, thresh=(160, 254))
    # get dark shadows area to exclude them from s-channel thresholding as regions with possible problems of detection
    shadow_area = layer_threshold(l_channel, kernel_size=kernel_size, thresh=config.shadow_threshold)
    # get sobel thresholds for s-channel
    # calculating defferent thresholds to L-channel of image
    s_gradx = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=kernel_size, thresh=(30, 150))
    s_grady = abs_sobel_thresh(s_channel, orient='y', sobel_kernel=kernel_size, thresh=(70, 150))
    s_mag_binary = mag_thresh(s_channel, sobel_kernel=9, thresh=(20, 100))
    s_dir_binary = dir_threshold(s_channel, sobel_kernel=15, thresh=(0, 1.3))
    s_mag_and_dir_binary = s_mag_binary & s_dir_binary
    # Stack s-channel and sobel magnitude\direction images to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(
        ((s_mag_binary & s_dir_binary) | s_gradx, shadow_area, (l_mag_binary & l_dir_binary))) * 255
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(l_channel)
    combined_binary[((s_binary == 1) | (((s_mag_binary == 1) & (s_dir_binary == 1)) | (s_gradx == 1)) | (
            (l_mag_binary == 1) & (l_dir_binary == 1))) & (shadow_area == 0)] = 1

    # warping image using transformation matrix M
    binary_warped = cv2.warpPerspective(combined_binary, config.M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
    color_warped = cv2.warpPerspective(color_binary, config.M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
    # drawing calibration lines on warped image
    # cv2.line(warped, (350, 0), (350, 720), color=[0, 0, 255], thickness=2)
    # cv2.line(warped, (910, 0), (910, 720), color=[0, 0, 255], thickness=2)

    cv2.imwrite("combined_binary.jpg",color_binary)

    warped_lines_drawn, left_line, right_line = find_lines(binary_warped, left_line, right_line)

    if visualise:
        S_L_channels, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        S_L_channels.tight_layout()
        ax1.imshow(s_channel, cmap='inferno')
        ax1.set_title('s-channel', fontsize=50)
        ax2.imshow(l_channel, cmap="inferno")
        ax2.set_title('l_channel', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plot, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        plot.tight_layout()
        ax1.imshow(s_mag_and_dir_binary, cmap='inferno')
        ax1.set_title('s_mag_binary - dir_binary', fontsize=50)
        ax2.imshow(l_mag_binary & l_dir_binary, cmap="inferno")
        ax2.set_title('l_mag & l_dir', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        #plt.show()


        combined, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        combined.tight_layout()
        ax1.imshow(color_binary)
        ax1.set_title('color_binary', fontsize=50)
        ax2.imshow(combined_binary,cmap="inferno")
        ax2.set_title('combined_binary', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        detected_lines, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        detected_lines.tight_layout()
        ax1.imshow(color_warped)
        ax1.set_title('color_warped', fontsize=50)
        ax2.imshow(warped_lines_drawn, cmap="inferno")
        ax2.set_title('warped_lines_drawn', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        #plt.show()

    #left_R_curve = left_line.rad_curv
    #right_R_curve = right_line.rad_curv
    if config.frame_number % config.avg_curv_frames ==0:
        config.curv=avg_curv(left_line,right_line,config.avg_curv_frames)

    # left_curverad = ((1+(2*left_fit_coefs[0]*y_eval+left_fit_coefs[1])**2)**1.5)/(2*np.abs(left_fit_coefs[0]))
    # right_curverad = ((1+(2*right_fit_coefs[0]*y_eval+right_fit_coefs[1])**2)**1.5)/(2*np.abs(right_fit_coefs[0]))

    # print ("Curvature",left_R_curve,' m',right_R_curve, ' m')

    # print ("Offset",offset," m")

    # applying line coefs smoothing using bicubic spline for each coefficient in x = a*y**2+b*y+c representation
    if (len(left_line.good_frames) > 5):

        # checking if we have enough frames
        if config.interp_len > len(left_line.good_frames):
            interp_len = len(left_line.good_frames)

        # flipping good frames list for corect work of spline interpolation, it requires ascending list of x values
        frame_grid = (left_line.good_frames[1:][-config.interp_len:])
        # print (left_line.good_frames)
        # extracting polyfit coeffs for last interp_len good fits
        left_a = (left_line.recent_fits[1:, 0][-config.interp_len:])
        right_a = (right_line.recent_fits[1:, 0][-config.interp_len:])
        left_b = (left_line.recent_fits[1:, 1][-config.interp_len:])
        right_b = (right_line.recent_fits[1:, 1][-config.interp_len:])
        left_c = (left_line.recent_fits[1:, 2][-config.interp_len:])
        right_c = (right_line.recent_fits[1:, 2][-config.interp_len:])
        a_left_spline = splrep(frame_grid, left_a, s=config.spline_smoothing)
        a_right_spline = splrep(frame_grid, right_a, s=config.spline_smoothing)
        b_left_spline = splrep(frame_grid, left_b, s=config.spline_smoothing)
        b_right_spline = splrep(frame_grid, right_b, s=config.spline_smoothing)
        c_left_spline = splrep(frame_grid, left_c, s=config.spline_smoothing)
        c_right_spline = splrep(frame_grid, right_c, s=config.spline_smoothing)
        # saving smoothened coeffs for drawing
        left_fit_coefs = [splev(config.frame_number, a_left_spline), splev(config.frame_number, b_left_spline),
                          splev(config.frame_number, c_left_spline)]
        right_fit_coefs = [splev(config.frame_number, a_right_spline), splev(config.frame_number, b_right_spline),
                           splev(config.frame_number, c_right_spline)]

    else:
        # if number of frames is less than 5 than we can't apply spline smoothing and use original values to draw line polygon
        left_fit_coefs = left_line.best_fit
        right_fit_coefs = right_line.best_fit

    ####TODO add error handling if line not detected
    if left_line.detected & right_line.detected:
        #workaround to disable smoothing if line is properly detedcted
        left_fit_coefs = left_line.current_fit
        right_fit_coefs = right_line.current_fit
        # offset calculation
        offset = (left_line.line_base_pos + right_line.line_base_pos) / 2
        # Highligting line in image
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        overlay_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # TODO draw expected line
        ploty = np.mgrid[0:binary_warped.shape[0]]

        left_fitx = np.uint16((left_fit_coefs[0] * ploty ** 2 + left_fit_coefs[1] * ploty + left_fit_coefs[2]))
        right_fitx = np.uint16((right_fit_coefs[0] * ploty ** 2 + right_fit_coefs[1] * ploty + right_fit_coefs[2]))
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(overlay_warp, np.int_([pts]), (0, 255, 0))
        # higlite line pixels
        overlay_warp[left_line.ally, left_line.allx] = [255, 0, 0]
        overlay_warp[right_line.ally, right_line.allx] = [0, 0, 255]
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(overlay_warp, config.Minv, (image.shape[1], image.shape[0]))
        # add placeholder for debug text
        cv2.fillPoly(newwarp, np.int_(config.debug_text_poly), (0, 255, 0))
        # Combine the result with the original image

        lines_drawn = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
        overlay_text = "Frame number %5i" % config.frame_number + " Curvature %8.2f" % (config.curv) + "m   offset %8.2fm" % (offset)
        if visualise:
            warps, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            ax1.imshow(overlay_warp)
            ax1.set_title('overlay_warp', fontsize=50)
            ax2.imshow(warped_lines_drawn)
            ax2.set_title('warped_lines_drawn', fontsize=50)
    else:
        overlay_text = "Frame number %5i" % config.frame_number + " Lines not detected"
        lines_drawn = np.copy(undistorted)
        if (left_fit_coefs is not None)&(right_fit_coefs is not None) :
            warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
            overlay_warp = np.dstack((warp_zero, warp_zero, warp_zero))
            ploty = np.mgrid[0:binary_warped.shape[0]]
            left_fitx = np.uint16((left_fit_coefs[0] * ploty ** 2 + left_fit_coefs[1] * ploty + left_fit_coefs[2]))
            right_fitx = np.uint16((right_fit_coefs[0] * ploty ** 2 + right_fit_coefs[1] * ploty + right_fit_coefs[2]))
            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(overlay_warp, np.int_([pts]), (0, 255, 0))

            # Warp the blank back to original image space using inverse perspective matrix (Minv)
            newwarp = cv2.warpPerspective(overlay_warp, config.Minv, (image.shape[1], image.shape[0]))
            # add placeholder for debug text

            cv2.fillPoly(newwarp, np.int_(config.debug_text_poly), (0, 255, 0))
            # Combine the result with the original image
            lines_drawn = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
            if visualise:
                warps, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
                ax1.imshow (overlay_warp)
                ax1.set_title('overlay_warp', fontsize=50)
                ax2.imshow(warped_lines_drawn)
                ax2.set_title('warped_lines_drawn', fontsize=50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(lines_drawn, overlay_text, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    if visualise:
        result, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        detected_lines.tight_layout()
        #undistorted=cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
        ax1.imshow(undistorted)
        #result_drawn = cv2.cvtColor(lines_drawn, cv2.COLOR_BGR2RGB)
        ax1.set_title('original image', fontsize=50)
        ax2.imshow(lines_drawn, cmap="inferno")
        ax2.set_title('processed image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
    return lines_drawn, left_line, right_line

