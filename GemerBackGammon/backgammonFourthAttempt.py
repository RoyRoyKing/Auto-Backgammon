import cv2 as cv
import numpy as np


#resize function 
def resize (img, scale = 0.4):
    dimensions = [int(img.shape[1] * scale), int(img.shape[0] * scale)]
    
    return cv.resize(img, dimensions, interpolation=cv.INTER_AREA)

#get board from file location
board = resize(cv.imread('BoardImages/New Project (1).jpg'), 0.5)
cv.imshow('og board', board)

#hsv
hsvBoard = cv.cvtColor(board, cv.COLOR_BGR2HSV)
cv.imshow('hsv board', hsvBoard)



#green thresholds
lower_green = np.array([35, 25, 25])
upper_green = np.array([90, 255,255])

#mask image
mask = cv.inRange(hsvBoard, lower_green, upper_green)
cv.imshow('mask', mask)

cv.waitKey(0)

#masked image
res = cv.bitwise_and(board,board, mask= mask)
cv.imshow('masked og image', res)

#blur
blur = cv.GaussianBlur(res, (1,1), 0)
cv.imshow('gaussian blurred board', blur)

#gray
gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
cv.imshow('gray masked', gray)

cv.waitKey(0)

#detect circles
circles = cv.HoughCircles(gray, 
                   cv.HOUGH_GRADIENT, 1, 20, param1 = 50,
               param2 = 30, minRadius = 1, maxRadius = 100)
print(circles)



cv.waitKey(0)