import cv2
import sys

parser = sys.argv
src = cv2.VideoCapture(parser[1])
if int(parser[3]) > src.get(cv2.CAP_PROP_FRAME_COUNT):
    parser[3] = src.get(cv2.CAP_PROP_FRAME_COUNT)
out = cv2.VideoWriter((parser[1].split('.'))[0]+'_{}-{}frames.'.format(int(parser[2]),int(parser[3]))+(parser[1].split('.'))[1], cv2.VideoWriter_fourcc(*'XVID'), int(src.get(cv2.CAP_PROP_FPS)), (int(src.get(cv2.CAP_PROP_FRAME_WIDTH)),int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))))
src.set(cv2.CAP_PROP_POS_FRAMES, int(parser[2]))
for i in range(int(parser[3])):
    ret, frame = src.read()
    if not ret: 
      break
    out.write(frame)
src.release()
out.release()
cv2.destroyAllWindows() 