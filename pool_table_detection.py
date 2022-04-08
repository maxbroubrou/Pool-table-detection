# STEP 1 : Detect the pool table as a blob (inspired by https://stackoverflow.com/questions/41138000/fit-quadrilateral-tetragon-to-a-blob)
# STEP 2 : Detection of the 6 main lines surrounding the blob (using cv2.ximgproc.createFastLineDetector())
# STEP 3 : get the pool table 4 coordinates that are the lines intersection points

import cv2
import math
import numpy as np

# read and resize image
img = cv2.imread("images/background.jpg")
img_copy = img.copy()
img = cv2.resize(img, (640, 360))
cv2.imshow("STEP 0 - Original Image", img)


# ---------------- STEP 1 ----------------

# blur image
gaussian_blur = cv2.GaussianBlur(img, (7,7), 0)

# apply HSV mask to highlight green
hsv = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)

hsv_lower = (46,10,10) # lower values for green mask
hsv_higher = (86,255,255) # max values for green mask

mask = cv2.inRange(hsv, hsv_lower, hsv_higher)
green = cv2.bitwise_and(gaussian_blur, gaussian_blur, mask = mask)
cv2.imshow("STEP 1 - Green Table", green)

# image processing to get a thresed blob
gray = cv2.cvtColor(green, cv2.COLOR_RGB2GRAY)
ret,thresh = cv2.threshold(gray,50,255,0)
kernel = np.ones((5,5), np.uint8)
dilate = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow("STEP 1 bis - Blob", dilate)


# ---------------- STEP 2 ----------------

main_line_number = 6
lines = []

class Line:
    def __init__(self, x1 : np.float32, y1 : np.float32, x2 : np.float32, y2 : np.float32) -> None:
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.length = math.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)

        self.a = (y2 - y1) / (x2 - x1)
        self.b = y1 - self.a * x1
    
    def __str__(self) -> str:
        str = (f'Coeff : {self.a} and Line : A({self.x1}, {self.y1}) to B({self.x2}, {self.y2})')
        return str


fld = cv2.ximgproc.createFastLineDetector().detect(dilate)
for line in fld:
    new_line = Line(line[0][0],line[0][1], line[0][2], line[0][3])
    lines.append(new_line)

lines.sort(key = lambda x: x.length, reverse=True) #sort lines by length

# Draw the line for visualisation
for line in lines[:main_line_number]:
    cv2.line(img, (line.x1, line.y1), (line.x2, line.y2), (255, 0, 255), 5)

cv2.imshow("STEP 2 - the 6 main lines detected", img)


# ---------------- STEP 3 ----------------

intersection_points = []
index_to_remove = []

def line_intersection(line1, line2): # source : https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


for i in range(len(lines[:main_line_number])):
    for j in range(i+1,len(lines[:main_line_number])):
        if (abs(lines[i].a - lines[j].a) > 0.5): # avoid the calculation of almost parallel lines
            intersection_points.append(line_intersection(((lines[i].x1,lines[i].y1), (lines[i].x2,lines[i].y2)),
            ((lines[j].x1, lines[j].y1), (lines[j].x2,lines[j].y2))))

# keep intersection points on the image size
intersection_points = [(int(i[0]), int(i[1])) for i in intersection_points if (i[0] >= 0 and i[1] >= 0 and i[0] <= img.shape[1] and i[1] <= img.shape[0])]

# remove too close intersection points
for i in range(len(intersection_points)):
    for j in range(i+1,len(intersection_points)):
        if(math.sqrt((intersection_points[j][0] - intersection_points[i][0])**2 + (intersection_points[j][1] - intersection_points[i][1])**2) < 100):
            index_to_remove.append(i)

intersection_points = [intersection_points[a] for a in range(len(intersection_points)) if a not in index_to_remove]

# Draw the circles for visualisation
for i in intersection_points:
    cv2.circle(img, (i[0],i[1]), 6, (2,0,255),-1)

cv2.imshow("STEP 3 - The 4 pool coordinates", img)
print(intersection_points)

cv2.waitKey(0)
cv2.destroyAllWindows()