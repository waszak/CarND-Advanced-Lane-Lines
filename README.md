

## Writeup Template


---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply gausian blur to distoreted images.
* Use sobel filter and S channel color filter to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistored_straight_lines1.jpg "Undistorted"
[image7]: ./output_images/distored_straight_lines1.jpg "Distorted"

[image2]: ./output_images/distored.jpg "Distorted chessboard"
[image8]: ./output_images/undistored.jpg "udistorted chessboard"

[image3]: ./output_images/color_img.jpg "Binary example"
[image9]: ./output_images/warped_img.jpg "Warped image"

[image4]: ./output_images/undist.jpg "Undistorted"

[image5]: ./output_images/warped_line.jpg "Fit Visual"
[image6]: ./output_images/test6.jpg "Output"
[video1]: ./test_videos_output/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the third code cell of the IPython notebook located in "./examples/AdvancedLaneLines.ipynb" (or in lines # through # of the file called `some_file.py`).  
First I precompute points (0,0,0),(1,0,0)....(9,6,0). Then I load files from calibration folder. For each of them I applay gray scale and try to find (9,6) corners. If they are then I remember maping between precomputed corners and detected corners.
Finally I use cv2 function calibrate cambera to use that mapings to perform calibration. Finally I can just use returned matrix to undistort images
``` python
class Camera:
    def __init__(self, nx=9, ny=6):
        self.nx=nx
        self.ny=ny

        # we fill obj_p with coordinates (0,0,0),(1,0,0)(2,0,0)...(nx-1,ny-1,0)
        self.obj_p = np.zeros((nx*ny,3), np.float32)
        k = 0
        for i in range(0,ny):
            for j in range(0,nx):
                self.obj_p[k,0] = j
                self.obj_p[k,1] = i
                k+=1    
                
    def calibrate(self, path, draw=False):
        files = glob.glob(path)
        obj_points = []
        img_points = []


        for filename in files:
            image = cv2.imread(filename)
            gray = grayscale(image)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret:
                obj_points.append(self.obj_p)
                img_points.append(corners)
                if draw:
                    cv2.drawChessboardCorners(image, (nx, ny) , corners, ret)
                    plt.imshow(image)

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None) 
    def undistort(self, img):
        return  cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
```
![alt text][image2] 
![alt text][image8] 
### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
Distorted
![alt text][image1] 
Undistorted
![alt text][image7]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. Step is cell 9

```python3
class BinaryFilter:
    def filter(self, img, mode='sobel'):
        if mode =='sobel':
            return self.sobel_filter(img)
        raise NotImplementedError('Mode is not implemented'+ mode)

    def sobel_filter(self, img, s_thresh=(170, 255), sx_thresh=(20, 100)):
        img = np.copy(img)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        
    
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) 
        abs_sobelx = np.absolute(sobelx) 
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
        
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        
        combined = np.zeros_like(s_channel)
        combined[(s_binary == 1) | (sxbinary == 1)] = 1 # 
        #color_binary for debuging
        color_binary = np.dstack(( np.zeros_like(sxbinary), combined, combined)) * 255#sx
        return combined, color_binary
```

![alt text][image3] 

![alt text][image9]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is included in cell 3

```python
    def birds_eye(self, img):
        shape = image.shape
        IMAGE_H = shape[0]
        IMAGE_W = shape[1]
        offset = 100
        src = np.float32([[450, 500], [850, 500],
                      [IMAGE_W, IMAGE_H], [0, IMAGE_H]])
        dst = np.float32([[0, 0], [IMAGE_W, 0],
                      [IMAGE_W, IMAGE_H], [0, IMAGE_H]])

        M = cv2.getPerspectiveTransform(src, dst) 
        Miv= cv2.getPerspectiveTransform(dst, src) 
        warped_img = self.warp_perspective(img,(IMAGE_W, IMAGE_H), M)
        #cropping
        warped_img[650:720,:,:]=0
        return warped_img, M, Miv
```


I verified that my perspective worked on test image.


![alt text][image4] 

![alt text][image9]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Cell 9 and 10 method find_lane_pixels, search_around_poly, fit_poly. I use first windows search and then i use serach around poly to for next frame. I also use avg of last 20 frames to makes line more smoother.

```python3
from collections import deque
class Lane:
    def __init__(self, shape):
        self.ym_per_pix = 30/720 
        self.xm_per_pix = 3.7/700 
        self.curvature = 0
        self.ploty = np.linspace(0, shape[0]-1, shape[0])
        self.fit = None
        self.fitx = None
        self.y = None
        self.pts = np.array([])
        self.queue_fit = deque()

    
    #https://en.wikipedia.org/wiki/Radius_of_curvature
    
    def measure_curvature_pixels(self):
        y_eval = np.max(self.y)
        fit = np.polyfit(self.ploty * self.ym_per_pix, self.fitx * self.xm_per_pix , 2)
        first_derivative = (2*fit[0]*y_eval*self.ym_per_pix + fit[1])
        first_derivative_square = first_derivative**2
        second_derivatie = 2*fit[0]
        curvature =  ((1 + first_derivative_square)**1.5 ) / np.absolute(second_derivatie)
        self.curvature = int(curvature)
        
    def fit_poly(self):
        self.fit = np.polyfit(self.y, self.x, 2)
        if len(self.queue_fit)==20:
            self.queue_fit.pop()
        for x in self.queue_fit:
            self.fit[0] +=x[0]
            self.fit[1] +=x[1]
            self.fit[2] +=x[2]
        
        self.fit[0] /= (len(self.queue_fit)+1)
        self.fit[1] /= (len(self.queue_fit)+1)
        self.fit[2] /= (len(self.queue_fit)+1)
        self.queue_fit.appendleft(self.fit)
        self.fitx = self.fit[0]*self.ploty**2 + self.fit[1]*self.ploty + self.fit[2]
        
            
    def offset(self):
        y_eval = np.max(self.y)
        return self.fit[0] * y_eval**2 + self.fit[1] * y_eval + self.fit[2]
    
    def find_lane_pixels(self, out_img, shape, current, nonzerox, nonzeroy):
        # HYPERPARAMETERS
        nwindows = 9
        margin = 60
        minpix = 50
        window_height = np.int(shape[0]//nwindows)
        
        inds = []
        for window in range(nwindows):
            win_y_low = shape[0] - (window+1)*window_height
            win_y_high = shape[0] - window*window_height
            win_low = current - margin
            win_high = current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_low,win_y_low),
            (win_high,win_y_high),(0,255,0), 2) 
            
            # Identify the nonzero pixels in x and y within the window #
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_low) &  (nonzerox < win_high)).nonzero()[0]

            # Append these indices to the lists
            inds.append(good_inds)

            if len(good_inds) > minpix:
                current = np.int(np.mean(nonzerox[good_inds]))

        lane_inds = np.concatenate(inds)
        
        if any(lane_inds):
            self.x = nonzerox[lane_inds]
            self.y = nonzeroy[lane_inds] 
    
    def search_around_poly(self, nonzerox, nonzeroy, binary_warped, out_img, window_img, color=[255,0,0]):
        margin = 50
        fit = self.fit
        lane_inds = ((nonzerox > (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + 
                        fit[2] - margin)) & (nonzerox < (fit[0]*(nonzeroy**2) + 
                        fit[1]*nonzeroy + fit[2] + margin)))
        
        if not any(lane_inds):
            print("line not detected")
            return False
        self.x = nonzerox[lane_inds]
        self.y = nonzeroy[lane_inds] 
    
        # Fit new polynomials
        self.fit_poly()
        fitx = self.fitx
        ploty = self.ploty


        # Color in left and right line pixels
        out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = color
    
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()

        line_window1 = np.array([np.transpose(np.vstack([fitx-margin, ploty]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx+margin, 
                                  ploty])))])
        line_pts = np.hstack((line_window1, line_window2))
  
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))
        return True
    
        
    def draw(self, max_y, out_img, color, flip=False):
        fitx = np.array(self.fitx)#[np.array(self.ploty) > max_y]
        y = np.array(self.ploty)#[np.array(self.ploty) > max_y]
        #we need to flip points for fill polygon
        if flip:
            pts = np.array([(np.transpose(np.vstack([fitx, y])))])
        else:
            pts = np.array([np.flipud(np.transpose(np.vstack([fitx, y])))])
            
        if pts.any():
            self.pts = pts
        
        cv2.polylines(out_img, np.int_([self.pts]),
                      isClosed=False, color=color, thickness=50)
        return self.pts
```

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in cell 9 and 10. I used formula for calculating radius of curvature for 2nd degree polynomial.
To calculate position of car i calculate left and right "offset"(by offset i mean value of in y_max point) then take avg. Center should be arround 640 pixel so i substract value and with this i get which side is correct.

```python3 
    class Lane:
        def __init__(self, shape):
            self.ym_per_pix = 30/720 
            self.xm_per_pix = 3.7/700 
        
        def measure_curvature_pixels(self):
            y_eval = np.max(self.y)
            fit = np.polyfit(self.ploty * self.ym_per_pix, self.fitx * self.xm_per_pix , 2)
            first_derivative = (2*fit[0]*y_eval*self.ym_per_pix + fit[1])
            first_derivative_square = first_derivative**2
            second_derivatie = 2*fit[0]
            curvature =  ((1 + first_derivative_square)**1.5 ) / np.absolute(second_derivatie)
            self.curvature = int(curvature)

        def offset(self):
            y_eval = np.max(self.y)
            return self.fit[0] * y_eval**2 + self.fit[1] * y_eval + self.fit[2]
    
    class LaneDetection:    
        def measure_curvature_pixels(self):
            self.left_lane.measure_curvature_pixels()
            self.right_lane.measure_curvature_pixels()


            left_offset = self.left_lane.offset()
            right_offset = self.right_lane.offset()
            center_offset = ((left_offset + right_offset)/2)-640

            offset_real_world = center_offset * self.left_lane.xm_per_pix
            self.position = round(offset_real_world, 2)
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines cell 9 and 10 in draw_lane method

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_videos_output/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipline fails at harder_challenge_video because of shadows and too bright areas. I have a lot issues with hard_challenge because lane sometimes dissapers. I was thinking about using one lane and translating it by vector to find other one. I didn't went in this direction because I wasn't sure if this is always a case.