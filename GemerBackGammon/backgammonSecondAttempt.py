import cv2 as cv

#resize function 
def resize (img, scale = 0.4):
    dimensions = [int(img.shape[1] * scale), int(img.shape[0] * scale)]
    
    return cv.resize(img, dimensions, interpolation=cv.INTER_AREA)

#get board from file location
board = resize(cv.imread('BoardImages/Board 6.jpg'), 0.4)
cv.imshow('og board', board)

#turn BGR to grayscale
gray = cv.cvtColor(board, cv.COLOR_BGR2GRAY)
cv.imshow('gray board', gray)

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
blur = cv.GaussianBlur(cont, (7, 7), 0)

#find edges
canny = cv.Canny(blur, 100, 200)
cv.imshow('board edges', canny)

#find contours
contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contsBoard = board.copy()
cv.drawContours(contsBoard, contours, -1, (0,255,0), 3)

cv.imshow('board with all contours', contsBoard)

#find corners
corners = cv.cornerHarris(grayCon,2,3,0.04)
corners = cv.dilate(corners, None)
corsBoard = board.copy()
corsBoard[corners>0.01*corners.max()]=[255,0,0]
cv.imshow('board with corners', corsBoard)

#hold images until keypress
cv.waitKey(0)

