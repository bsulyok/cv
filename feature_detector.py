import cv2
import sys
from matplotlib import pyplot as plt

parser = sys.argv
#orb = cv2.ORB_create(200, 1.2, 8, 15, 0, 2, cv2.ORB_HARRIS_SCORE, 31, 20)
img = cv2.imread(parser[1])
orb = cv2.ORB_create()
kp = orb.detect(img)
cv2.drawKeypoints(img, kp, img)

plt.imshow(img)
plt.show()
