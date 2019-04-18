import cv2
from matplotlib import pyplot as plt
import sys

parser = sys.argv
src = cv2.imread(parser[1])

orb = cv2.ORB_create(200, 1.2, 8, 15, 0, 2, cv2.ORB_HARRIS_SCORE, 31, 20)
kp = orb.detect(src)
img = src.copy()
for marker in kp:
	img = cv2.drawMarker(img, tuple(int(i) for i in marker.pt), color=(0, 255, 0))

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray, 200, 0.01, 30)
img2 = src.copy()
for i in range(len(corners)):
    x,y = corners[i].ravel()
    cv2.circle(img2, (x,y), 5, (0,0,255-255/len(corners)*i), 5, 8, 0)
cv2.imshow("orb", img)
cv2.imshow("feat", img2)
cv2.waitKey()