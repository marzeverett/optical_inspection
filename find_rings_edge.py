

import numpy as np
import cv2 as cv
#Resources:
#https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
#https://docs.opencv.org/4.x/d8/d23/classcv_1_1Moments.html
 

def nothing(x):
    pass

def contours(img_name):
    #img_name = f"imgs/ponytail_1.jpg"
    key = ord("r")
    #cv.namedWindow('controls')
    #cv.createTrackbar("lower", 'controls', 0, 225, nothing)
    #cv.createTrackbar("upper", 'controls', 0, 225, nothing)
    while key != ord("s"):
        still = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
        #https://www.geeksforgeeks.org/image-resizing-using-opencv-python/
        still = cv.resize(still, (0, 0), fx = 0.1, fy = 0.1)
        og_img = still.copy()
        img = still
        #Gaussian Blur
        img = cv.GaussianBlur(img, (7,7), 0)
        # lower = int(cv.getTrackbarPos('lower', 'controls'))
        # upper = int(cv.getTrackbarPos('upper', 'controls'))

        lower = 66
        upper = 185 

        img = cv.Canny(img, lower, upper)
        contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        #Get biggest contour, taken directly from below: 
        #https://stackoverflow.com/questions/44588279/find-and-draw-the-largest-contour-in-opencv-on-a-specific-color-python
        areas = [cv.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        single_contour=contours[max_index]

        #Taken directly from here: https://pyimagesearch.com/2016/02/01/opencv-center-of-contour/ 
        moments = cv.moments(single_contour)
        center_point_x = int(moments["m10"]/moments["m00"])
        center_point_y = int(moments["m01"]/moments["m00"])

        cv.drawContours(og_img, [single_contour], -1, (255, 0, 0), 3)
        cv.circle(og_img, (center_point_x, center_point_y), 2, (255, 255, 255), -1)
        img = og_img

        cv.imshow("Image", img)
        key = cv.waitKey(0)
        return img 

for i in range(1, 7):
    img_name = f"imgs/ponytail_{i}.jpg"
    cimg = contours(img_name)
    #Save it 
    cv.imwrite(f"processed_imgs/ponytail_{i}_edge.jpg", cimg)

#Could improve this by grabbing area for the contours and going from there 
