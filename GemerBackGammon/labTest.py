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
#cv.imshow('l component lab board', l)
#cv.imshow('a component lab board', a)
#cv.imshow('b component lab board', b)

clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
cv.imshow('contrasted l component lab board', cl)

limg = cv.merge((cl,a,b))
final = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
cv.imshow('final merged contrast', final)

cv.waitKey(0)