import cv2
import math
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

feature_num, feature_quality, feature_distance = 200, 0.01, 30

parser = sys.argv
vidname, vidformat = parser[1].split('.')
src = cv2.VideoCapture("{}.{}".format(vidname, vidformat))
frame_num, fps = int(src.get(cv2.CAP_PROP_FRAME_COUNT)), src.get(cv2.CAP_PROP_FPS)
horiz, vert = src.get(cv2.CAP_PROP_FRAME_WIDTH), src.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_size = (int(horiz), int(vert))

orb = cv2.ORB_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

ret, prev = src.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

last_transform, trajectory = [], []
a, x, y = 0, 0, 0
for i in tqdm(range(frame_num)):
    ret, cur = src.read()
    if not ret:
        break
    cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    cur_features = cv2.goodFeaturesToTrack(cur_gray, feature_num, feature_quality, feature_distance)
    cur_features, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_features, cur_features)
    prev_tracked_features = prev_features[np.where(status == 1)[0]]
    cur_tracked_features = cur_features[np.where(status == 1)[0]]
    trans, ret2 = cv2.estimateAffinePartial2D(prev_tracked_features, cur_tracked_features)
    if trans is None:
        trans = last_transform
    last_transform = trans
    x, y, a = x + trans[0][2], y + trans[1][2], a + np.arctan2(trans[1][0], trans[0][0])
    trajectory.append([-x, -y, -a])
    prev, prev_gray, prev_features = cur, cur_gray, cur_features

src.set(cv2.CAP_PROP_POS_FRAMES, 1)
out = cv2.VideoWriter("{}_stab.{}".format(vidname, vidformat), cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)

for trans in tqdm(trajectory):
    ret, cur = src.read()
    if not ret:
        break
    cur = cv2.warpAffine(cur, np.array([[math.cos(trans[2]), -math.sin(trans[2]), trans[0]], [math.sin(trans[2]), math.cos(trans[2]), trans[1]]]), frame_size)
    out.write(cur)
out.release()
src.release()

with open("trajectory.txt", "w") as output:
    for traj in trajectory:
        output.write("{}\n".format(traj))

plt.plot(trajectory)
plt.show()
