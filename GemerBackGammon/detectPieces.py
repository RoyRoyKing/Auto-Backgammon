from cv2 import boxPoints, rectify3Collinear
from findShapes import *
from backgammonCalibration import *
import cv2 as cv
import numpy as np
import imutils
import matplotlib.pyplot as plt

#calibrate board with empty image
triSegments = calibrate(cv.imread('BoardImages/Custom Board 2.jpeg'), True)

#locate board in current positing
board = locateBoard(cv.imread('BoardImages/Custom Board 8.jpeg'), True)

#board for drawing boxes
#boxesBoard = board.copy()

#detecting pieces
pieces = detectPieces(board, True, True)

#print(pieces)

#for p in pieces:
    #print(f'{pieces.index(p)} {pieceColor(board, p)}')
    
pieceColor(board, pieces[3])

drawBoxes(board, triSegments, (0, 255, 0), 2)

cv.imshow('segments', board)
cv.waitKey(0)