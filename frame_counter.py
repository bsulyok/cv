import cv2
import math
import argparse
import sys

parser = sys.argv
src = cv2.VideoCapture(parser[1])
print(src.get(cv2.CAP_PROP_FRAME_COUNT))
src.release()
cv2.destroyAllWindows() 