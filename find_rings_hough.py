#Copied pretty much directly from the opencv_docs
#https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html 
#Also a doc linl to the function 
#https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d 

import numpy as np
import cv2 as cv
 

#Takes an image name, does the transform - press key to advance the loop 
def hough_circle_transform(img_name):
    img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
    #https://www.geeksforgeeks.org/image-resizing-using-opencv-python/
    img = cv.resize(img, (0, 0), fx = 0.1, fy = 0.1)
    assert img is not None, "file could not be read, check with os.path.exists()"
    img = cv.medianBlur(img,5)
    cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
    
    circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,
    param1=50,param2=30,minRadius=0,maxRadius=0)
    
    #Returns circles  - two center coordinates of where the circle is located
    #Plus radius of the circle 
    circles = np.uint16(np.around(circles))
    largest_radius = 0
    coord_1 = 0
    coord_2 = 0
    #Get the largest radius of the circle (inefficient way of doing this, admittedly)
    for i in circles[0,:]:
        if i[2] > largest_radius:
            coord_1 = i[0]
            coord_2 = i[1]
            largest_radius = i[2]
        
    # draw the outer circle
    cv.circle(cimg,(coord_1,coord_2), largest_radius,(0,255,0),2)
    # draw the center of the circle
    cv.circle(cimg,(coord_1, coord_2),2,(0,0,255),3)
    
    cv.imshow('detected circles',cimg)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return cimg


for i in range(1, 7):
    img_name = f"imgs/ponytail_{i}.jpg"
    cimg = hough_circle_transform(img_name)
    #Save it 
    cv.imwrite(f"processed_imgs/ponytail_{i}_hough.jpg", cimg)
