import cv2
import numpy as np
import sys
import math
import cv3

parser = sys.argv
frame_count = 100
diff = int(parser[2])
src = cv2.imread(parser[1])
out = cv2.VideoWriter("fluctuate.mkv", cv2.VideoWriter_fourcc(*'XVID'), 20, (600, 600))

for i in range(frame_count):
    p = diff*(2*(i%2)-1)
    trans = [p, 0, 0]
    trans2 = np.array([[math.cos(trans[2]), -math.sin(trans[2]), trans[0]], [math.sin(trans[2]), math.cos(trans[2]), trans[1]]])
    cur = cv2.warpAffine(src, trans2, (600, 600))
    #cur = cv3.rotate(src,p)
    out.write(cur)
out.release()
