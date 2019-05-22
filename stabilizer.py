import cv2
import math
import numpy as np
import sys
from tqdm import tqdm

# Reading the input video
parser = sys.argv
vidname, vidformat = parser[1].split('.')
src = cv2.VideoCapture("{}.{}".format(vidname, vidformat))
frame_num, fps = int(src.get(cv2.CAP_PROP_FRAME_COUNT)), src.get(cv2.CAP_PROP_FPS)
horiz, vert = src.get(cv2.CAP_PROP_FRAME_WIDTH), src.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_size = (int(horiz), int(vert))

# Initializing the descriptor extractor and matcher
detector = cv2.ORB_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

# Reading the first frame
ret, prev = src.read()
gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
kp_prev, des_prev = detector.detectAndCompute(gray_prev, None)

# Iterating over the next frame, estimating the transformation between frame and previous frame
last_trans, trajectory = None, []
a, x, y = 0, 0, 0
for frames in tqdm(range(frame_num)):
    ret, cur = src.read()
    if not ret:
        break
    gray_cur = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    kp_cur, des_cur = detector.detectAndCompute(gray_cur, None)
    matches = matcher.match(des_prev, des_cur)
    kp_1 = [kp_prev[M.queryIdx].pt for M in matches]
    kp_2 = [kp_cur[M.trainIdx].pt for M in matches]
    inliers = np.empty((2,3))
    trans, ret = cv2.estimateAffinePartial2D(np.array(kp_1), np.array(kp_2), inliers, cv2.RANSAC, 3, 2000, 0.99) 
    if trans is None:
        trans = last_trans
    last_trans = trans
    x, y, a = x + trans[0][2], y + trans[1][2], a + np.arctan2(trans[1][0], trans[0][0])
    trajectory.append([-x, -y, -a])
    gray_prev, kp_prev, des_prev = gray_cur, kp_cur, des_cur

# Removing large-scale translations, leaving only the fluctuations
fluct_trajectory, radius = [], 20
for i in range(len(trajectory)):
    sum_x, sum_y, sum_a, c = 0, 0, 0, 0
    for j in range(-radius, radius):
        if 0 <= i+j < len(trajectory):
            sum_x, sum_y, sum_a = sum_x + trajectory[i+j][0], sum_y + trajectory[i+j][1], sum_a + trajectory[i+j][2]
            c+=1
    fluct_trajectory.append([trajectory[i][0]-sum_x/c, trajectory[i][1]-sum_y/c, trajectory[i][2]-sum_a/c])

# Applying correction
src.set(cv2.CAP_PROP_POS_FRAMES, 1)
out = cv2.VideoWriter("{}_stab.{}".format(vidname, vidformat), cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)
for traj in tqdm(fluct_trajectory):
    ret, cur = src.read()
    if not ret:
        break
    cur = cv2.warpAffine(cur, np.array([[math.cos(traj[2]), -math.sin(traj[2]), traj[0]], [math.sin(traj[2]), math.cos(traj[2]), traj[1]]]), frame_size)
    out.write(cur)
out.release()
src.release()
