import numpy as np
import cv2
from matplotlib import pyplot as plt

img2 = cv2.imread('data/smaplepic.png',0)

orb = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500)
kp2 = orb.detect(img2)

img = img2.copy()
for marker in kp2:
	img = cv2.drawMarker(img, tuple(int(i) for i in marker.pt), color=(0, 255, 0))

plt.figure()
plt.imshow(img)
plt.show()