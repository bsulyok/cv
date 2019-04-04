import cv2
import math
import numpy as np
import sys


trans = [5,6,30]
transform = [[math.cos(trans[2]),-math.sin(trans[2]),trans[0]],[math.sin(trans[2]),math.cos(trans[2]),trans[1]]]


print(transform)