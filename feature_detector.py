import cv2
from matplotlib import pyplot as plt
import sys

parser = sys.argv
src = cv2.imread(parser[1])
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

kp = cv2.goodFeaturesToTrack(gray, 200, 0.01, 30)

img = src.copy()

for marker in kp:
    x, y = marker.ravel()
    img = cv2.drawMarker(img, (x, y), (0, 255, 0))

cv2.imshow(img)
