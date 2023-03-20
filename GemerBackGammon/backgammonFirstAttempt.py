from tkinter import W, PhotoImage

from cv2 import rectify3Collinear
from findShapes import *
import cv2 as cv
import numpy as np
import imutils
import matplotlib.pyplot as plt

#resize function 
def resize (img, scale = 0.4):
    dimensions = [int(img.shape[1] * scale), int(img.shape[0] * scale)]
    
    return cv.resize(img, dimensions, interpolation=cv.INTER_AREA)

#get board from file location
board = resize(cv.imread('BoardImages/Custom Board 2.jpeg'), 0.4)
cv.imshow('og board', board)

#grayscale
gray = cv.cvtColor(board, cv.COLOR_BGR2GRAY)
cv.imshow('gray board', gray)

#gaussian blur
blur = cv.GaussianBlur(gray, (3, 3), 0)
cv.imshow('gaussian blurred board', blur)

#edge detection
canny = cv.Canny(gray, 100, 200)
cv.imshow('board edges', canny)

#contour board copy
conBoard = board.copy()

#finding contours
contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(conBoard, contours, -1, (0,255,0), 3)
cv.imshow('contour board', conBoard)

cv.waitKey(0)

#board rectangles
recBoard = board.copy()


#finding rectangles
rectangles = findShapes(contours, 4)
print(len(rectangles))

#new points board
pBoard = board.copy()

#corners
print(board.shape[0])
print(board.shape[1])
imgCorners = np.array([[[0, board.shape[0]]], [[board.shape[1], board.shape[0]]], [[board.shape[1], 0]], [[0,0]]])

#draw corner points
for p in imgCorners:
    pBoard = cv.circle(pBoard, (p[0][0], p[0][1]), radius=5, color=(0, 0, 255), thickness=-1)
    print(p)

cv.imshow('corners', pBoard)
#print(imgCorners)

#drawing rectangles
cv.drawContours(recBoard, rectangles, -1, (0,255,0), 3)
cv.imshow('rectangles board', recBoard)

cv.waitKey(0)

#finding potential board rectangles
screenArea = board.shape[1] * board.shape[0]
boardRecs = [rec for rec in rectangles if cv.contourArea(rec) > 0.1 * screenArea]

#contour areas
contourAreas = [cv.contourArea(rec) for rec in boardRecs]

#finding correct board rectangle
boardRec = boardRecs[contourAreas.index(min(contourAreas))]

#contour background
conBoard = board.copy()

#draw correct board on image
cv.drawContours(conBoard,[boardRec],-1,(0,255,0),3)
cv.imshow('board rectangle', conBoard)

cv.waitKey(0)

#quadrilateral background
quadBoard = board.copy()

#approximating to quadrilateral
peri = cv.arcLength(boardRec, True)
boardCorners = cv.approxPolyDP(boardRec, 0.04 * peri, True)

print(boardCorners)

#drawing quadrilateral
cv.polylines(quadBoard, [boardCorners], True, (0,0,255), 1, cv.LINE_AA)
cv.imshow('quadrilateral board', quadBoard)

#board to warp
warpBoard = board.copy()
squareBlank = np.zeros((board.shape[0], board.shape[0], 3), dtype=np.uint8)
pBlank = squareBlank.copy()

#square board corners
squareCorners = np.array([[[0, squareBlank.shape[0]]], [[squareBlank.shape[1], squareBlank.shape[0]]], [[squareBlank.shape[1], 0]], [[0,0]]])

cv.waitKey(0)
cv.destroyAllWindows()


#aligning image to board only
hom, status = cv.findHomography(boardCorners, squareCorners)
onlyBoard = cv.warpPerspective(warpBoard, hom, (squareBlank.shape[1], squareBlank.shape[0]))
#drawing warped image
cv.imshow('only board', onlyBoard)

cv.waitKey(0)

#board to grayscale
grayBoard = cv.cvtColor(onlyBoard,cv.COLOR_BGR2GRAY)
cv.imshow('gray board', grayBoard)

#eliminating noise
blurOnlyBoard = cv.GaussianBlur(onlyBoard, (3, 3), 0)
cv.imshow('blurred board', blurOnlyBoard)

#board to hsv
hsvBoard = cv.cvtColor(blurOnlyBoard, cv.COLOR_BGR2HSV)
cv.imshow('hsv board', hsvBoard)

#red hsv values
lowerRed1 = np.array([0, 100, 20])
upperRed1 = np.array([10, 255, 255])

lowerRed2 = np.array([160,100,20])
upperRed2 = np.array([179,255,255])

#red mask
lowerRedMask = cv.inRange(hsvBoard, lowerRed1, upperRed1)
upperRedMask = cv.inRange(hsvBoard, lowerRed2, upperRed2)
 
redMask = lowerRedMask + upperRedMask
cv.imshow('red mask', redMask)

#masked board
redMasked = cv.bitwise_and(onlyBoard,onlyBoard, mask= redMask)
cv.imshow('red masked board', redMasked)

#find red edges
cannyRedBoard = cv.Canny(redMasked, 100, 200)
cv.imshow('edges red board', cannyRedBoard)

#connecting edges
def closeEdges(val):
            # create a kernel based on trackbar input
            kernel = np.ones((val,val))
            # do a morphologic close
            res = cv.morphologyEx(cannyRedBoard,cv.MORPH_CLOSE, kernel)
            # display result
            cv.imshow("Result", res)
            return res

closed = closeEdges(3)

cv.imshow('closed edges', closed)

cv.waitKey(0)

#new board for drawing contours
onlyConBoard = onlyBoard.copy()

#finding contours
redContours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(onlyConBoard, redContours, -1, (0,255,0), 3)

print(type(redContours))

cv.imshow('red contours', onlyConBoard)

#filtering small contours
''' for cnt in redContours:
    if cv.contourArea(cnt) < 50:
        redContours.remove(cnt) '''
    

#finding triangles
triangles = findShapes(redContours, 3)

print('triangles length:' + str(len(triangles)))
print(triangles)

#new board for drawing triangles
triBoard = onlyBoard.copy()

#drawing triangles
cv.drawContours(triBoard, triangles, -1, (0,255,0), 3)
cv.imshow('triangles board', triBoard)

#new board for drawing bounding boxes
boxOnlyBoard = onlyBoard.copy()

#all bounding boxes
boundings = []

#bounding boxes of contours
for cnt in redContours:
    x,y,w,h = cv.boundingRect(cnt)
    cv.rectangle(boxOnlyBoard,(x,y),(x+w,y+h),(0,255,0),2)
    boundings.append((x,y,w,h))
    boundings.append((0, 0, 0, 0))

print(boundings)

for bnd in boundings:
    boxOnlyBoard = cv.circle(boxOnlyBoard, (bnd[0], bnd[1]), radius = 5, color=(0, 0, 255), thickness=-1)
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(boxOnlyBoard, str(boundings.index(bnd)), (bnd[0] + 10, bnd[1] + 10), font, 1, (0, 255, 0), 2, cv.LINE_AA)

cv.imshow('boxes', boxOnlyBoard)

for i in range(len(boundings)):
    if i % 4 == 1:
        boundings.insert(i, (boundings[i + 1][0], boundings[i + 1][1] + boundings[i + 1][3], boundings[i + 1][2], boundings[i + 1][3]))
        cv.rectangle(boxOnlyBoard,(boundings[i][0],boundings[i][1]),(boundings[i][0]+boundings[i][2],boundings[i][1]+boundings[i][3]),(0,0,255),2)
        
cv.imshow('more boxes', boxOnlyBoard)
cv.waitKey(0)