## **Advanced Lane Finding Project**


---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Check that lane finding is correct (lines are paraller, distance between lines is about what we expect in real life
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
* Handle line detection errors and draw estimated lane boundaries if lane detection didn't succseed

[//]: # (Image References)

[image1]: ./Writeup/Undistorted.png "Undistorted"
[image2]: ./Writeup/undistorted.jpg "Road Transformed"
[image3]: ./Writeup/combined_binary.jpg "Binary Example"
[image4]: ./Writeup/warp.png "Warp Example"
[image5]: ./Writeup/warped_lines_drawn.jpg "Fit Visual"
[image6]: ./Writeup/result.jpg "Output"
[image7]: ./Writeup/554_Figure_1.png
[image8]: ./Writeup/project554.jpg
[image9]: ./Writeup/Histogram.png
[video1]: /Writeup/project_video.avi "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README   

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function calles `camera_calibrate`  in lines 7 through 53 of the file called `camera_calibrate.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
17 of 20 provided images were good for calibration (all chessboard cells were visible and recognized by cv2.findChessboardCorners)
I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

In order not to do camera calibration every run of image processing pipeline, after camera calibration, I saved mtx and dst matrices to file `mtx_dist_pickle.p` using `pickle` module

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

As mentioned earlier, I saved camera calibration matrices to `mtx_dist_pickle.p` and every run of my pipeline I load them in `config.py` file used for defining global variables and configuration constants.
First step in image processing is distortion correction, in my pipeline I do it in function `process_image()` function defined in `functions.py` lines 463-682. This is and example of distortion corrected image from project video.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I processed image converted to HSL colorspace. I combined magnitude and direction thresholds for L -layer, simple S-layer thresholding, magnitude and direction thresholds for S-layer and threshold for partial derivative along x-axis for S-layer  using functions `mag_thresh`, `dir_threshold`, `abs_sobel_thresh` and `layer_threshold` defined in `functions.py`. Also I masked out shadowed image parts with L-value less than `30` as not reliable sources for thresholding using `layer_threshold` function.
Result thresholded image is formed using this formula 
```python 
combined_binary[((s_binary == 1) | (((s_mag_binary == 1) & (s_dir_binary == 1)) | (s_gradx == 1)) | (
            (l_mag_binary == 1) & (l_dir_binary == 1))) & (shadow_area == 0)] = 1 
```
From hereon I will not mention where function is defined as all of them are stored in `functions.py` file
This is an example of thresholded binary image:
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
For perspective transform I call cv2.warpPerspective within my `process_image` funcion
Warp transform matrices are calculated in `config.py` and referenced through pipeline as global parameters
I chose this values and they are defined as hyperparameters in `config.py` file
```python
src = np.float32([[190, 720], [563, 470], [720, 470], [1100, 720]])
dst = np.float32([[350, 720], [350, 0], [940, 0], [940, 720]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 190, 720     | 350, 720        | 
| 563, 470      | 350, 0     |
| 720, 470    | 940, 0      |
| 1100, 720     | 940, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After warp transform I indentified lane-line pixels using "sliding window" and "search around last frame polyline" methods implemented in `sliding_window` and `search_around_poly` function 
If last frame was not good or this is first frame we use "sliding window" to identify lane line pixels then we do poly fit with second power poly using `NumPy` `polyfit` function. And then we do some sanity check for identified polyfit in `sanity_check` function (lines sould be of similar parameters (i.e. parallel) and distance between lines shold be about lane width)
If we know poly-line fit from last frame then we do lane line search within certain margin from this line. If snaity check for for this lines fail then sliding window fallback is implemented.
If lines pass sanity check then frame number is added to good frames array and polyfit coefficients are added to historical array. Both of them are used to provide extrapolations if line detection failed for certain frame. Bicubic spline extrapolation is used in this pipeline. 

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Radius of curvature is calculated with `radius_of_curv` function. 
```python
scld_fit = [fit_coefs[0] * config.xm_per_pix / (config.ym_per_pix ** 2),fit_coefs[1] * config.xm_per_pix,fit_coefs[2] * config.xm_per_pix]

rad_curv = ((1 + (2 * scld_fit[0] * y_eval * config.ym_per_pix + scld_fit[1]) ** 2) ** 1.5) / (2 * np.abs(scld_fit[0]))
```
And car position in line is calculated within  image processing function `process_image` using `find_lines` function results
```python
#left_line.fitx is an array of x-values for polyfit line ( (fitx[y];y) make points of polyfit line)
find_lines()
...
    y_eval = (binary_warped.shape[0] - 1)
    left_line.line_base_pos = (left_line.fitx[y_eval] - binary_warped.shape[1] // 2) * config.xm_per_pix
    right_line.line_base_pos = (right_line.fitx[y_eval] - binary_warped.shape[1] // 2) * config.xm_per_pix
...
image_process()
...
    offset = (left_line.line_base_pos + right_line.line_base_pos) / 2
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 594 through 655 in image processing function `process_image`. Code is rather big due to implementation of spline extrapolation and various errors handling  
```python
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

```
Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](Input videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
* Hardcoding or manually setting thresholds for image isn't very scalable solution. Almost all images in provided videos are well-lit and  if weather changes to cloudy or sun sets to dusk, parameters I chosen will not work as well as they did there. I think that some image metering or exposure information from camera might help to choose parameters automatically.
* Another drawback of my pipeline are manually set warp transform parameters. This means warp transform will work well only if car pitch doesn't change. If pitch changes, parallel lines become not so parallel after warp transform. Here is an example:
![alt text][image8]
![alt text][image7]
Automatic horison detection might help to solve this problem. Horison can be detected using pitch sensor of a car or some image processing. And using horison location we can adjust warp transform for every frame avoiding this problem.
* When we speak about handling frames with no lines detected, bicubic spline extrapolation did quite well, but for line smoothing it didn't work quite well as lines became oscillating. Probably increasing smoothing parameter or increasing\decreasing historical depth will help.
* Slidning window line detection method is simple and elegant and yet it can provide false line detections if we have some marks on asphalt like in challenge video. Improving it with binding windows to expected distance between lines or creating multipass sliding window working not only with maximal values on histogram, but on second order maximums
![alt text][image9]