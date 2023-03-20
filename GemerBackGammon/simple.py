import cv2 as cv

#resize function 
def resize (img, scale = 0.4):
    dimensions = [int(img.shape[1] * scale), int(img.shape[0] * scale)]
    
    return cv.resize(img, dimensions, interpolation=cv.INTER_AREA)

#get board from file location
board = resize(cv.imread('BoardImages/Board 3.jpg'), 0.4)
cv.imshow('og board', board)

#grayscale
gray = cv.cvtColor(board, cv.COLOR_BGR2GRAY)
cv.imshow('gray board', gray)

#l*a*b
lab = cv.cvtColor(board, cv.COLOR_BGR2LAB)
l, a, b = cv.split(lab)
cv.imshow('l component lab board', l)
cv.imshow('a component lab board', a)
cv.imshow('b component lab board', b)

#gaussian blur
gauss_blur = cv.GaussianBlur(gray, (1, 1), 0)
cv.imshow('gaussian blurred board', gauss_blur)

#bilateral filter
bilateral_blur = cv.bilateralFilter(gray, 5, 30, 30)
#cv.imshow('bilateral blurred board', bilateral_blur)

#edge detection
canny = cv.Canny(gray, 100, 200)
#cv.imshow('board edges', canny)

#finding contours
contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(board, contours, -1, (0,255,0), 3)
#cv.imshow('contoured board', board)

cv.waitKey(0)