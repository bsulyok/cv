import cv2
import sys
import numpy as np

parser = sys.argv
src = cv2.VideoCapture(parser[1])
out_name = (parser[1].split('.'))[0]+'_feat.'+(parser[1].split('.'))[1]
out_dim = (int(src.get(cv2.CAP_PROP_FRAME_WIDTH)), int(src.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'XVID'), int(src.get(cv2.CAP_PROP_FPS)), out_dim)
orb = cv2.ORB_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

ret, prev = src.read()
gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
kp_prev, des_prev = orb.detectAndCompute(gray_prev, None)

while(src.isOpened()):
    ret, cur = src.read()
    if not ret:
        break
    gray_cur = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    kp_cur, des_cur = orb.detectAndCompute(gray_cur,  None)
    matches = matcher.match(des_prev, des_cur)
    
    kp_1 = [kp_prev[M.queryIdx].pt for M in matches]
    kp_2 = [kp_cur[M.trainIdx].pt for M in matches]
    trans, ret = cv2.estimateAffinePartial2D(np.array(kp_1), np.array(kp_2), None, cv2.RANSAC, 3, 2000, 0.99) 
    inlier = np.array((len(ret)), cv2.KeyPoint)
    for k in range(len(ret)):
        if ret[k]:
            cv2.drawMarker(cur, (int(kp_1[k][0]), int(kp_1[k][1])), (0, 255, 0))
    #cv2.drawKeypoints(cur,inlier, cur)
    out.write(cur)
src.release()
out.release()
cv2.destroyAllWindows() 
