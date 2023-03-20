import cv2 as cv
import numpy as np
import math

## THIS IS VER 3 WITH A DIFFERENT CANNY APPROACH

#get image and resize to uniform size
img = cv.imread("images/im6.jpg")
img = cv.resize(img, (1920//2, 1080//2))

cv.imshow("bgr", img)

#get hsv image
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
hue, sat, val = cv.split(hsv)
cv.imshow("hue", hue)
cv.imshow("sat", sat)
cv.imshow("val", val)

# min_hue, max_hue = 10, 40
# hue_sum = 0
# hue_count = 0
# for p in hue.flatten():
#   if p <= max_hue and p >= min_hue:
#     hue_sum += p
# avg_hue = hue_sum//hue_count #TODO: make this work

# #blur and denoise
# blur = cv.GaussianBlur(hsv, (5, 5), 2)

#generate result image
res = img.copy()

#filter by color
min_color = np.array([10, 0, 150])
max_color = np.array([27,255,255])
color_mask = cv.inRange(hsv, min_color, max_color)

cv.imshow("mask", color_mask)
cv.waitKey(0)

#apply canny edges
canny = cv.Canny(img, 250, 400)
cv.imshow("canny", canny)

#find contours of canny
canny_cnts, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(color_mask, canny_cnts, -1, (0), thickness=-1)
cv.drawContours(color_mask, canny_cnts, -1, (0), thickness=2)

cv.imshow("mask", color_mask)
cv.waitKey(0)

#find contours of mask
cnts, _ = cv.findContours(color_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)

#filter contours by area
MIN_AREA = 200
REALLY_MIN_AREA = 40
small_cnts = [c for c in cnts if cv.contourArea(c) < MIN_AREA and cv.contourArea(c) >= REALLY_MIN_AREA]
cnts = [c for c in cnts if cv.contourArea(c) >= MIN_AREA]

color_cnts = cv.cvtColor(color_mask, cv.COLOR_GRAY2BGR)
cv.drawContours(color_cnts, cnts, -1, (0, 0, 255))
cv.drawContours(color_cnts, small_cnts, -1, (255, 0, 0))
cv.imshow("mask", color_cnts)
cv.waitKey(0)

bboxes = []
for c in cnts:
    x, y, w, h = cv.boundingRect(c)
    bboxes.append([(x,y), (x+w, y+h)])

cannied_bboxes = []
MAX_WHITE_RATIO = 0.07
for b in bboxes:
    x1, y1 = b[0]
    x2, y2 = b[1] #get bounding rect of contour
    white = 0
    for x in range(x1, x2):
        for y in range(y1, y2):
            if canny[y, x] == 255: white+=1 #counnt number of white pixels in bounding rect
    
    print(f"{white} / {x2-x1}*{y2-y1} = {white / (x2-x1)*(y2-y1)}")
    if white / ((x2-x1)*(y2-y1)) > MAX_WHITE_RATIO: continue #skip rects that have too much white (edges)
    cannied_bboxes.append(b)
    
    #cv.rectangle(final_mask, (x1, y1), (x2, y2), (255), thickness=-1) #draw rects onto final mask
    #cv.rectangle(res, (x1, y1), (x2, y2), (0, 0, 255), thickness=2) #draw rects onto image

bboxes = cannied_bboxes

#join small selections that are close to big ones
MAX_CNT_DIST = 70
new_cnts = []
for small in small_cnts:
    sx, sy, sw, sh = cv.boundingRect(small) #get bounding rect of contour
    s_center = (sx+sw//2, sy+sh//2)
    s_radius = (sh+sw)//2

    closest_big = None
    closest_dist = MAX_CNT_DIST
    for i in range(len(bboxes)):
        big = bboxes[i]
        bx, by = b[0]
        bw, bh = (b[1][0] - bx, b[1][1] - by) #get bounding rect of contour
        b_center = (bx+bw//2, by+bh//2)
        b_radius = (bh+bw)//2

        x_dist, y_dist = abs(b_center[0]-s_center[0]), abs(b_center[1]-s_center[1])
        dist = math.hypot(x_dist, y_dist) #distance between edges of bboxes
        
        # temp = img.copy()
        # cv.drawContours(temp, (small), -1, (255, 0, 0), 2)
        # cv.drawContours(temp, (big), -1, (0, 0, 255), 2)
        # cv.imshow("bbox joining", temp)
        # print(dist)
        # cv.waitKey(0)

        if dist <= closest_dist: #and dist > b_radius + s_radius:
          closest_big = i
          closest_dist = dist

    if closest_big is not None:
      #add points to shared bbox to calculate later
      
      bboxes[closest_big].append((sx, sy))
      bboxes[closest_big].append((sx+sw, sy+sh))

      new_cnts.append(small)
      print("joined cnt")

for c in new_cnts: cnts.append(c)

final_mask = np.zeros(hue.shape, np.uint8) #empty binary size of image

# bbox_img = img.copy()
bboxes = [b for b in bboxes if len(b) != 0]
for i in range(len(bboxes)):
    box = bboxes[i]
    
    min_x, min_y, max_x, max_y = math.inf, math.inf, -math.inf, -math.inf
    for p in box:
        min_x = min(min_x, p[0])
        min_y = min(min_y, p[1])
        max_x = max(max_x, p[0])
        max_y = max(max_y, p[1])

    bboxes[i] = (min_x, min_y, max_x, max_y)
    cv.rectangle(final_mask, (min_x, min_y), (max_x, max_y), (255), -1)

# cv.imshow("bboxes", bbox_img)
# cv.waitKey(0)

cv.imshow("final mask", final_mask)
cnts, _ = cv.findContours(final_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(res, cnts, -1, (0, 0, 255), 2)

cv.imshow("res", res)
cv.waitKey(0)