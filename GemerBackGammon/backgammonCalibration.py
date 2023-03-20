from cv2 import blur, rectify3Collinear
from findShapes import *
import cv2 as cv
import numpy as np
import imutils
import matplotlib.pyplot as plt

NUM_POINTS = 24

#resize function 

def locateBoard(img, showSteps = False):
    board = resize(img, 0.4)
    
    
    

    #grayscale
    gray = cv.cvtColor(board, cv.COLOR_BGR2GRAY)
    

    #gaussian blur
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    

    #edge detection
    canny = cv.Canny(gray, 100, 200)
    

    #contour board copy
    conBoard = board.copy()

    #finding contours
    contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(conBoard, contours, -1, (0,255,0), 3)
    
    #board rectangles
    recBoard = board.copy()


    #finding rectangles
    rectangles = findShapes(contours, 4)
    #print(len(rectangles))

    #new points board
    pBoard = board.copy()

    #corners
    #print(board.shape[0])
    #print(board.shape[1])
    imgCorners = np.array([[[0, board.shape[0]]], [[board.shape[1], board.shape[0]]], [[board.shape[1], 0]], [[0,0]]])

    #draw corner points
    for p in imgCorners:
        pBoard = cv.circle(pBoard, (p[0][0], p[0][1]), radius=5, color=(0, 0, 255), thickness=-1)
        #print(p)

    #drawing rectangles
    cv.drawContours(recBoard, rectangles, -1, (0,255,0), 3)

    #finding potential board rectangles
    screenArea = board.shape[1] * board.shape[0]
    boardRecs = [rec for rec in rectangles if cv.contourArea(rec) > 0.1 * screenArea]

    #print(board.shape[1])

    #contour areas
    contourAreas = [cv.contourArea(rec) for rec in boardRecs]

    #finding correct board rectangle
    boardRec = boardRecs[contourAreas.index(min(contourAreas))]

    #contour background
    conBoard = board.copy()

    #draw correct board on image
    cv.drawContours(conBoard,[boardRec],-1,(0,255,0),3)

    #quadrilateral background
    quadBoard = board.copy()

    #approximating to quadrilateral
    peri = cv.arcLength(boardRec, True)
    boardCorners = cv.approxPolyDP(boardRec, 0.04 * peri, True)

    #print(boardCorners)

    #fixing orientation
    boardCorners = sortCorners(boardCorners)

    #print(boardCorners)

    #drawing quadrilateral
    cv.polylines(quadBoard, [boardCorners], True, (0,0,255), 1, cv.LINE_AA)
    

    #board to warp
    warpBoard = board.copy()
    squareBlank = np.zeros((board.shape[0], board.shape[0], 3), dtype=np.uint8)
    pBlank = squareBlank.copy()

    #square board corners
    squareCorners = np.array([[[0, squareBlank.shape[0]]], [[squareBlank.shape[1], squareBlank.shape[0]]], [[squareBlank.shape[1], 0]], [[0,0]]])

    
    #aligning image to board only
    hom, status = cv.findHomography(boardCorners, squareCorners)
    onlyBoard = cv.warpPerspective(warpBoard, hom, (squareBlank.shape[1], squareBlank.shape[0]))
    #drawing warped image
    

    if showSteps: 
        cv.imshow('og board', board)
        cv.waitKey(0)
        cv.imshow('gray board', gray)
        cv.waitKey(0)
        cv.imshow('gaussian blurred board', blur)
        cv.waitKey(0)
        cv.imshow('board edges', canny)
        cv.waitKey(0)
        cv.imshow('contour board', conBoard)
        cv.waitKey(0)
        cv.imshow('corners', pBoard)
        cv.waitKey(0)
        cv.imshow('rectangles board', recBoard)
        cv.waitKey(0)
        cv.imshow('board rectangle', conBoard)
        cv.waitKey(0)
        cv.imshow('quadrilateral board', quadBoard)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imshow('only board', onlyBoard)
        cv.waitKey(0)

    return onlyBoard

def calibrate(img, showSteps = False):
    #find board in image
    onlyBoard = locateBoard(img, showSteps)

    #board to grayscale
    grayBoard = cv.cvtColor(onlyBoard,cv.COLOR_BGR2GRAY)
    

    #eliminating noise
    blurOnlyBoard = cv.GaussianBlur(onlyBoard, (3, 3), 0)
    

    #board to hsv
    hsvBoard = cv.cvtColor(blurOnlyBoard, cv.COLOR_BGR2HSV)
    

    #red hsv values
    lowerRed1 = np.array([0, 100, 20])
    upperRed1 = np.array([10, 255, 255])

    lowerRed2 = np.array([160,100,20])
    upperRed2 = np.array([179,255,255])

    #red mask
    lowerRedMask = cv.inRange(hsvBoard, lowerRed1, upperRed1)
    upperRedMask = cv.inRange(hsvBoard, lowerRed2, upperRed2)
    
    redMask = lowerRedMask + upperRedMask
    

    #masked board
    redMasked = cv.bitwise_and(onlyBoard,onlyBoard, mask= redMask)
    

    #find red edges
    cannyRedBoard = cv.Canny(redMasked, 100, 200)
    

    #connecting edges


    closed = closeEdges(3, cannyRedBoard)

    

    cv.waitKey(0)

    #new board for drawing contours
    onlyConBoard = onlyBoard.copy()

    #finding contours
    redContours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(onlyConBoard, redContours, -1, (0,255,0), 3)

    #print(type(redContours))

    

    #filtering small contours
    ''' for cnt in redContours:
        if cv.contourArea(cnt) < 50:
            redContours.remove(cnt) '''
        

    #finding triangles
    triangles = findShapes(redContours, 3)

    #print('triangles length:' + str(len(triangles)))
    #print(triangles)

    #new board for drawing triangles
    triBoard = onlyBoard.copy()

    #drawing triangles
    cv.drawContours(triBoard, triangles, -1, (0,255,0), 3)
    

    #new board for drawing bounding boxes
    boxOnlyBoard = onlyBoard.copy()
    boxOnlyBoard2 = onlyBoard.copy()

    #all bounding boxes
    boundings = []

    #bounding boxes of contours
    for cnt in redContours:
        x,y,w,h = cv.boundingRect(cnt)
        #cv.rectangle(boxOnlyBoard2,(x,y),(x+w,y+h),(0,255,0),2)
        boundings.append((x,y,w,h))

    drawBoxes(boxOnlyBoard2, boundings, (0, 255, 0), 2)


    for bnd in boundings:
        boxOnlyBoard2 = cv.circle(boxOnlyBoard2, (bnd[0], bnd[1]), radius = 5, color=(0, 0, 255), thickness=-1)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(boxOnlyBoard2, str(boundings.index(bnd)), (bnd[0] + 10, bnd[1] + 10), font, 1, (0, 255, 0), 2, cv.LINE_AA)
        #print(str(boundings.index(bnd)))

    

    srtBoundings = [(0, 0, 0, 0)] * 24


    #organizing boundings
    for i in range(len(boundings)):
        if boundings[i][0] < onlyBoard.shape[0] / 2:
            srtBoundings[i] = boundings[i]
        else:
            #pTwelve = i + 12
            #print(str(i) + '      ' + str(boundings[i]))
            srtBoundings[i+12] =  boundings[i]
            #print(srtBoundings[i+12])
            #print(pTwelve)
            #print(str(i) + ': ' + str(srtBoundings))


    #finding black triangle boundings
    for i in range(len(srtBoundings)):
        if (i % 2 == 0 and i < 12) or (i % 2 == 1 and i > 12):
            if i == 0 or i == 6:
                srtBoundings[i] = (srtBoundings[i+1][0], srtBoundings[i+1][1] + srtBoundings[i+1][3], srtBoundings[i+1][2], srtBoundings[i+1][3])
            elif i == 23 or i == 17:
                srtBoundings[i] = (srtBoundings[i-1][0], srtBoundings[i-1][1] - srtBoundings[i-1][3], srtBoundings[i-1][2], srtBoundings[i-1][3])
            else:
                srtBoundings[i] = (srtBoundings[i+1][0], srtBoundings[i+1][1] + srtBoundings[i+1][3], srtBoundings[i+1][2], (srtBoundings[i-1][1] - (srtBoundings[i+1][1] + srtBoundings[i+1][3])))
            #cv.rectangle(boxOnlyBoard,(srtBoundings[i][0], srtBoundings[i][1]),(srtBoundings[i][0]+srtBoundings[i][2],srtBoundings[i][1]+srtBoundings[i][3]),(0,0,255),2)

    drawBoxes(boxOnlyBoard, srtBoundings, (0, 0, 255), 2)

    #print(srtBoundings)

    

    if showSteps: 
        cv.imshow('gray board', grayBoard)
        cv.waitKey(0)
        cv.imshow('blurred board', blurOnlyBoard)
        cv.waitKey(0)
        cv.imshow('hsv board', hsvBoard)
        cv.waitKey(0)
        cv.imshow('red mask', redMask)
        cv.waitKey(0)
        cv.imshow('red masked board', redMasked)
        cv.waitKey(0)
        cv.imshow('edges red board', cannyRedBoard)
        cv.waitKey(0)
        cv.imshow('closed edges', closed)
        cv.waitKey(0)
        cv.imshow('red contours', onlyConBoard)
        cv.waitKey(0)
        cv.imshow('triangles board', triBoard)
        cv.waitKey(0)
        cv.imshow('og boxes', boxOnlyBoard2)
        cv.waitKey(0)
        cv.imshow('more boxes', boxOnlyBoard)
        cv.waitKey(0)
        cv.destroyAllWindows()


    return srtBoundings


def detectPieces (squareBoard, showSteps = False, displayIndexes = False):

    #grayscale
    grayBoard = cv.cvtColor(squareBoard, cv.COLOR_BGR2GRAY)

    #blur
    blurBoard = cv.medianBlur(grayBoard,1)

    #board for drawing circles:
    crclBoard = squareBoard.copy()

    #circle size
    offset = 7
    radius = 17.5 / 455 * squareBoard.shape[0]
    minRadius = int(radius - offset)
    maxRadius = int(radius + offset)

    #find circles
    pCircles = cv.HoughCircles(blurBoard,cv.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=minRadius,maxRadius=maxRadius)

    #organizing circles
    pCircles = pCircles.tolist()[0]
    for crcl in pCircles:
        crcl[0] = int(crcl[0])
        crcl[1] = int(crcl[1])
        crcl[2] = int(crcl[2])

    #print (f'the list has {len(pCircles)} circles')
    #print(pCircles)

    #draw circles
    #print(pCircles[0])
    #drawShape(crclBoard, (1, 1.5, 3), 'circle', (255, 0, 0), 2)
    for circle in pCircles:
        drawShape(crclBoard, circle, 'circle', (255, 0, 0), 2)
        if displayIndexes:
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(crclBoard, str(pCircles.index(circle)), (circle[0], circle[1]), font, 1, (0, 255, 0), 2, cv.LINE_AA)

    if showSteps:
        cv.imshow('gray board', grayBoard)
        cv.waitKey(0)
        cv.imshow('blurred board', blurBoard)
        cv.waitKey(0)
        cv.imshow('pieces', crclBoard)
        cv.waitKey(0)
        #cv.destroyAllWindows()

    
    
    return pCircles

def detectBoard(segments, pieces):
    board = [(0, None)] * NUM_POINTS

    for p in pieces:
        for seg in segments:
            if isInBox(p, seg):
                board[segments.index(seg)][0] += 1

    return board
