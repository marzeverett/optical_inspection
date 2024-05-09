

import numpy as np
import cv2 as cv
#Resources:
#https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
#https://docs.opencv.org/4.x/d8/d23/classcv_1_1Moments.html
 

webcam = cv.VideoCapture(0)


def nothing(x):
    pass


def contours():
    #img_name = f"imgs/ponytail_1.jpg"
    key = ord("r")
    #cv.namedWindow('controls')
    #cv.createTrackbar("lower", 'controls', 0, 225, nothing)
    #cv.createTrackbar("upper", 'controls', 0, 225, nothing)
    while key != ord("s"):
        still = webcam.read()
        og_img = still[1].copy()
        #Convert to grayscale
        img = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)

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
        area_of_contour = cv.contourArea(single_contour)

        cv.drawContours(og_img, [single_contour], -1, (255, 0, 0), 3)
        cv.circle(og_img, (center_point_x, center_point_y), 2, (255, 255, 255), -1)
        #https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
        cv.putText(og_img, str(area_of_contour), (center_point_x, center_point_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
        img = og_img

        cv.imshow("Image", img)
        key = cv.waitKey(5)

contours()
#Could improve this by grabbing area for the contours and going from there 


# image = cv2.putText(image, 'OpenCV', org, font,  
#                    fontScale, color, thickness, cv2.LINE_AA) 