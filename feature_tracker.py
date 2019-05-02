import cv2
import sys


parser = sys.argv
src = cv2.VideoCapture(parser[1])
out_name = (parser[1].split('.'))[0]+'_feat.'+(parser[1].split('.'))[1]
out_dim = (int(src.get(cv2.CAP_PROP_FRAME_WIDTH)), int(src.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'XVID'), int(src.get(cv2.CAP_PROP_FPS)), out_dim)
orb = cv2.ORB_create()
while(src.isOpened()):
    ret, frame = src.read()
    if not ret:
        break
    kp = orb.detect(frame)
    kp, des = orb.compute(frame, kp)
    cv2.drawKeypoints(frame, kp, frame)
    out.write(frame)
src.release()
out.release()
cv2.destroyAllWindows() 
    
