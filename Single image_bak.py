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
    l_gradx = functions.abs_sobel_thresh(l_channel, orient='x', sobel_kernel=kernel_size, thresh=(40, 150))
    l_grady = functions.abs_sobel_thresh(l_channel, orient='y', sobel_kernel=kernel_size, thresh=(0, 40))
    l_mag_binary = functions.mag_thresh(l_channel, sobel_kernel=9, thresh=(70, 200))
    l_dir_binary = functions.dir_threshold(l_channel, sobel_kernel=15, thresh=(0, 1.3))
    # get simple s-layer threshold
    s_binary = functions.layer_threshold(s_channel, kernel_size=kernel_size, thresh=(160, 254))
    # get dark shadows area to exclude them from s-channel thresholding as regions with possible problems of detection
    shadow_area = functions.layer_threshold(l_channel, kernel_size=kernel_size, thresh=(0, 8))
    # get sobel thresholds for s-channel
    # calculating defferent thresholds to L-channel of image
    s_gradx = functions.abs_sobel_thresh(s_channel, orient='x', sobel_kernel=kernel_size, thresh=(30, 150))
    s_grady = functions.abs_sobel_thresh(s_channel, orient='y', sobel_kernel=kernel_size, thresh=(70, 150))
    s_mag_binary = functions.mag_thresh(s_channel, sobel_kernel=9, thresh=(20, 100))
    s_dir_binary = functions.dir_threshold(s_channel, sobel_kernel=15, thresh=(0, 1.3))
    s_mag_and_dir_binary = s_mag_binary & s_dir_binary
    # Stack s-channel and sobel magnitude\direction images to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(
        ((s_mag_binary & s_dir_binary) | s_gradx, np.zeros_like(l_channel), (l_mag_binary & l_dir_binary))) * 255
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(l_channel)
    combined_binary[((s_binary == 1) | (((s_mag_binary == 1) & (s_dir_binary == 1)) | (s_gradx == 1)) | (
            (l_mag_binary == 1) & (l_dir_binary == 1))) & (shadow_area == 0)] = 1

    # warping image using transformation matrix M
    binary_warped = cv2.warpPerspective(combined_binary, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
    color_warped = cv2.warpPerspective(color_binary, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
    # drawing calibration lines on warped image
    # cv2.line(warped, (350, 0), (350, 720), color=[0, 0, 255], thickness=2)
    # cv2.line(warped, (910, 0), (910, 720), color=[0, 0, 255], thickness=2)

    warped_lines_drawn, left_line, right_line = functions.find_lines(binary_warped, left_line, right_line)

    if visualise:
        #S_L_channels, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        #S_L_channels.tight_layout()
        #ax1.imshow(s_channel, cmap='inferno')
        #ax1.set_title('s-channel', fontsize=50)
        #ax2.imshow(l_channel, cmap="inferno")
        #ax2.set_title('l_channel', fontsize=50)
        #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        #plot, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        #plot.tight_layout()
        #ax1.imshow(s_mag_and_dir_binary, cmap='inferno')
        #ax1.set_title('s_mag_binary - dir_binary', fontsize=50)
        #ax2.imshow(l_mag_binary, cmap="inferno")
        #ax2.set_title('l_mag', fontsize=50)
        #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        # l_plot, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        # l_plot.tight_layout()
        # ax1.imshow(s_grady,cmap='binary_r')
        # ax1.set_title('s_grady', fontsize=50)
        # ax2.imshow(l_dir_binary,cmap="binary_r")
        # ax2.set_title('l_dir', fontsize=50)
        # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        # mag_plot, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        # mag_plot.tight_layout()
        # ax1.imshow(s_mag_and_dir_binary,cmap="binary_r")
        # ax1.set_title('s_mag_binary - dir_binary', fontsize=50)
        # ax2.imshow(l_gradx&l_grady,cmap="binary_r")
        # ax2.set_title('s_dir_binary', fontsize=50)
        # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        # combined, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        # combined.tight_layout()
        # ax1.imshow(color_binary)
        # ax1.set_title('color_binary', fontsize=50)
        # ax2.imshow(combined_binary,cmap="inferno")
        # ax2.set_title('combined_binary', fontsize=50)
        # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        detected_lines, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        detected_lines.tight_layout()
        ax1.imshow(color_warped)
        ax1.set_title('color_warped', fontsize=50)
        ax2.imshow(warped_lines_drawn, cmap="inferno")
        ax2.set_title('warped_lines_drawn', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
    left_R_curve = left_line.rad_curv
    right_R_curve = right_line.rad_curv

    # left_curverad = ((1+(2*left_fit_coefs[0]*y_eval+left_fit_coefs[1])**2)**1.5)/(2*np.abs(left_fit_coefs[0]))
    # right_curverad = ((1+(2*right_fit_coefs[0]*y_eval+right_fit_coefs[1])**2)**1.5)/(2*np.abs(right_fit_coefs[0]))

    # print ("Curvature",left_R_curve,' m',right_R_curve, ' m')

    # print ("Offset",offset," m")

    # applying line coefs smoothing using bicubic spline for each coefficient in x = a*y**2+b*y+c representation
    if (len(left_line.good_frames) > 5):

        # checking if we have enough frames
        if interp_len > len(left_line.good_frames):
            interp_len = len(left_line.good_frames)

        # flipping good frames list for corect work of spline interpolation, it requires ascending list of x values
        frame_grid = (left_line.good_frames[1:][-interp_len:])
        # print (left_line.good_frames)
        # extracting polyfit coeffs for last interp_len good fits
        left_a = (left_line.recent_fits[1:, 0][-interp_len:])
        right_a = (right_line.recent_fits[1:, 0][-interp_len:])
        left_b = (left_line.recent_fits[1:, 1][-interp_len:])
        right_b = (right_line.recent_fits[1:, 1][-interp_len:])
        left_c = (left_line.recent_fits[1:, 2][-interp_len:])
        right_c = (right_line.recent_fits[1:, 2][-interp_len:])
        a_left_spline = splrep(frame_grid, left_a, s=config.spline_smoothing)
        a_right_spline = splrep(frame_grid, right_a, s=config.spline_smoothing)
        b_left_spline = splrep(frame_grid, left_b, s=config.spline_smoothing)
        b_right_spline = splrep(frame_grid, right_b, s=config.spline_smoothing)
        c_left_spline = splrep(frame_grid, left_c, s=config.spline_smoothing)
        c_right_spline = splrep(frame_grid, right_c, s=config.spline_smoothing)
        # saving smoothened coeffs for drawing
        left_fit_coefs = [splev(frame_number, a_left_spline), splev(frame_number, b_left_spline),
                          splev(frame_number, c_left_spline)]
        right_fit_coefs = [splev(frame_number, a_right_spline), splev(frame_number, b_right_spline),
                           splev(frame_number, c_right_spline)]

    else:
        # if number of frames is less than 5 than we can't apply spline smoothing and use original values to draw line polygon
        left_fit_coefs = left_line.current_fit
        right_fit_coefs = right_line.current_fit

    ####TODO add error handling if line not detected
    if left_line.detected & right_line.detected:
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
        print (pts_left)
        print (pts_left.shape)
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(overlay_warp, np.int_([pts]), (0, 255, 0))
        # higlite line pixels
        overlay_warp[left_line.ally, left_line.allx] = [255, 0, 0]
        overlay_warp[right_line.ally, right_line.allx] = [0, 0, 255]
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(overlay_warp, Minv, (image.shape[1], image.shape[0]),flags=cv2.INTER_LINEAR)
        # add placeholder for debug text

        cv2.fillPoly(newwarp, np.int_(config.debug_text_poly), (0, 255, 0))
        # Combine the result with the original image
        lines_drawn = cv2.addWeighted(undistorted, 1, newwarp, 0.5, 0)
        overlay_text = "Frame number %5i" % config.frame_number + " Curvature %8.2f" % (
                (left_R_curve + right_R_curve) / 2) + "m   offset %8.2fm" % (offset)
    else:
        overlay_text = "Frame number %5i" % frame_number + " Lines not detected"
        lines_drawn = np.copy(undistorted)
        if (len(left_line.good_frames) > 5):
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
            newwarp = cv2.warpPerspective(overlay_warp, Minv, (image.shape[1], image.shape[0]),flags=cv2.INTER_LINEAR)
            # add placeholder for debug text
            cv2.fillPoly(newwarp, np.int_(debug_text_poly), (0, 255, 0))
            # Combine the result with the original image
            lines_drawn = cv2.addWeighted(undistorted, 1, newwarp, 0.5, 0)



    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(lines_drawn, overlay_text, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    if visualise:
        warps, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        ax1.imshow(overlay_warp)
        ax1.set_title('overlay_warp', fontsize=50)
        ax2.imshow(newwarp)
        ax2.set_title('newwarp', fontsize=50)
        plt.show()
        result, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        detected_lines.tight_layout()
        ax1.imshow(undistorted)
        ax1.set_title('original image', fontsize=50)
        ax2.imshow(lines_drawn, cmap="inferno")
        ax2.set_title('processed image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    return lines_drawn, left_line, right_line



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
img = mpimg.imread('.\extracted_images\project20.jpg')

# processing image
output, left_line, right_line = process_image(img, left_line, right_line,config.interp_len, True)

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