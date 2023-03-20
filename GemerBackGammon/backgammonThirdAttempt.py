import cv2 as cv
import numpy as np
import math

#resize function 
def resize (img, scale = 0.4):
    dimensions = [int(img.shape[1] * scale), int(img.shape[0] * scale)]
    
    return cv.resize(img, dimensions, interpolation=cv.INTER_AREA)

#get board from file location
board = resize(cv.imread('BoardImages/Board 6.jpg'), 0.6)
#cv.imshow('og board', board)

#grayscale
gray = cv.cvtColor(board, cv.COLOR_BGR2GRAY)
cv.imshow('gray board', gray)

#gaussian blur
blur = cv.GaussianBlur(gray, (3, 3), 0)
#cv.imshow('gaussian blurred board', blur)

#split BGR to l*a*b
lab = cv.cvtColor(board, cv.COLOR_BGR2LAB)
l, a, b = cv.split(lab)
#cv.imshow('l component lab board', l)
#cv.imshow('a component lab board', a)
#cv.imshow('b component lab board', b)

#contrast l component image
clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
cv.imshow('contrasted l component lab board', cl)

#merge contrasted component 
limg = cv.merge((cl,a,b))

#turn back from l*a*b to BGR
cont = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
cv.imshow('final merged contrast', cont)

#turn contrasted BGR to grayscale
grayCon = cv.cvtColor(cont, cv.COLOR_BGR2GRAY)
cv.imshow('gray contrasted board', grayCon)

#blur
blur = cv.GaussianBlur(grayCon, (3, 3), 0)

#edge detection
canny = cv.Canny(blur, 20, 200, None, 3)
cv.imshow('board edges', canny)

#colored canny
ccanny =cv.cvtColor(canny, cv.COLOR_GRAY2BGR)
cv.imshow('colored canny', ccanny)

#copies of canny
ccanny2 = ccanny.copy()
ccanny3 = ccanny.copy()



#finding lines --> hough lines transform
lines = cv.HoughLines(canny, 1, np.pi / 180, 150, None, 0, 0)

#finding lines --> probabilistic hough lines transform
linesP = cv.HoughLinesP(canny, 1, np.pi / 180, 50, None, 50, 20)

#drawing regular hough lines
if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(ccanny2, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
            
#drawing probabilistic hough lines
if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(ccanny3, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
            
cv.imshow("Source", board)
cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", ccanny2)
cv.imshow("Detected Lines (in red) - Probabilistic Hough Line Transform", ccanny3)
    
cv.waitKey(0)