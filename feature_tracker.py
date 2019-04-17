import cv2
import sys

cv2.ORB()
parser = sys.argv
src = cv2.VideoCapture(parser[1])
out = cv2.VideoWriter((parser[1].split('.'))[0]+'_feat.'+(parser[1].split('.'))[1], cv2.VideoWriter_fourcc(*'XVID'), int(src.get(cv2.CAP_PROP_FPS)), (int(src.get(cv2.CAP_PROP_FRAME_WIDTH)),int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))))
while(src.isOpened()):
    ret, frame = src.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB()
    kp = orb.detect(frame, None)
    kp, des = orb.compute(frame, kp)
    #frame = cv2.drawKeypoints(frame, kp, color=(0,255,0), flags = 0)
    #corners = cv2.goodFeaturesToTrack(gray, int(parser[2]), 0.01, 30)
    #for i in range(len(corners)):
    #    x,y = corners[i].ravel()
    #    cv2.circle(frame, (x,y), 5, (0,0,255-255/len(corners)*i), 5, 8, 0)
    out.write(frame)
src.release()
out.release()
cv2.destroyAllWindows() 
    