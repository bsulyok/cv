import cv2
import sys

parser = sys.argv
src = cv2.VideoCapture(parser[1])
out = cv2.VideoWriter((parser[1].split('.'))[0]+'_'+parser[2]+'fps.'+(parser[1].split('.'))[1], cv2.VideoWriter_fourcc(*'XVID'), int(parser[2]), (int(src.get(cv2.CAP_PROP_FRAME_WIDTH)),int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))))
while(src.isOpened()):
    ret, frame = src.read()
    if not ret: 
      break
    out.write(frame)
src.release()
out.release()
cv2.destroyAllWindows() 