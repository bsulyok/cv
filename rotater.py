import cv2
import cv3
import math
import sys

parser = sys.argv
vidname, vidformat = parser[1].split('.')
rotspeed = parser[2]
src = cv2.VideoCapture("{}.{}".format(vidname, vidformat))
diagonal = int(math.sqrt(src.get(cv2.CAP_PROP_FRAME_WIDTH)**2+src.get(cv2.CAP_PROP_FRAME_HEIGHT)**2))
out = cv2.VideoWriter("{}_rot{}.{}".format(vidname, rotspeed, vidformat), cv2.VideoWriter_fourcc(*'XVID'), src.get(cv2.CAP_PROP_FPS), (diagonal, diagonal))
angle = 0
while src.isOpened():
    ret, cur = src.read()
    if not ret:
        break
    cur = cv3.pad_rotate(cur, angle)
    cv2.imshow("win", cur)
    cv2.waitKey(2)
    out.write(cur)
    angle += 2
src.release()
out.release()
cv2.destroyAllWindows()
