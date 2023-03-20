import cv2 as cv
import numpy as np


def findShapes (contours, edgeNum):
    shapes = []
    for cnt in contours:
        epsilon = 0.02*cv.arcLength(cnt,True)
        approx = cv.approxPolyDP(cnt,epsilon,True)
        
        if len(approx) == edgeNum: 
            shapes.append(approx)
            
    return shapes

def sortCorners (cornersArr):
    corners = cornersArr.tolist()
    #print(corners)

    sums = [(crn[0][0] + crn[0][1]) for crn in corners]
    bRight = corners[sums.index(max(sums))]
    tLeft = corners[sums.index(min(sums))]

    #print('bottom right: ' + str(bRight))
    #print('top left: ' + str(tLeft))

    remaining = [crn for crn in corners if crn is not bRight and crn is not tLeft]
    remaining.sort()

    tRight = remaining[1]
    bLeft = remaining[0]

    #print('top right: ' + str(tRight))
    #print('bottom left: ' + str(bLeft))

    return np.asarray([tRight, bRight, bLeft, tLeft])

def resize (img, scale = 0.4):

    #dimensions = [int(img.shape[1] * scale), int(img.shape[0] * scale)]
    dimensions = [640, 480]

    return cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
    
def closeEdges(val, board):
    # create a kernel based on trackbar input
    kernel = np.ones((val,val))
    # do a morphologic close
    res = cv.morphologyEx(board,cv.MORPH_CLOSE, kernel)
    # display result
    cv.imshow("Result", res)
    return res

def drawShape(img, src, shape, color, width):
    shape = shape.lower()
    if shape == 'box' or shape == 'square' or shape == 'rectangle':
        cv.rectangle(img,(src[0], src[1]),(src[0]+src[2],src[1]+src[3]),color,width)
    elif shape == 'circle':
        cv.circle(img,(src[0],src[1]),src[2],color,width)

def drawBoxes(img, boxes, color, width):
    for box in boxes:
        drawShape(img, box, 'box', color, width)

def isInBox(pnt, box):
    if pnt[0] < box[0] + box[2] and pnt[0] > box[0] and pnt[1] < box[1] + box[3] and pnt[1] > box[1]:
        return True
    return False

def pieceColor(img, p):     #(x, y, radius)
    #find 10 points in piece and average color
    tPixels = []
    for i in range(7):
        tPixels.append(list(img[p[0] + i, p[1]]))
        tPixels.append(list(img[p[0] - i, p[1]]))
        tPixels.append(list(img[p[0], p[1] + i]))
        tPixels.append(list(img[p[0], p[1] - i]))
        
        drawShape(img, (p[0] + i, p[1], 1), 'circle', (0, 0, 0), 1)
        drawShape(img, (p[0] - i, p[1], 1), 'circle', (0, 0, 0), 1)
        drawShape(img, (p[0], p[1] + i, 1), 'circle', (0, 0, 0), 1)
        drawShape(img, (p[0], p[1] - i, 1), 'circle', (0, 0, 0), 1)
        
    
    avg = [0, 0, 0] #GRB
    print(tPixels)
    for pxl in tPixels:
        avg[0] += pxl[0]
        avg[1] += pxl[1]
        avg[2] += pxl[2]
        
        #print(avg)
        
    for i in range(len(avg)):
        avg[i] /= len(tPixels)
        avg[i] = int(avg[i])
    
    
    return avg
