import cv2
import math
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = sys.argv
vidname, vidformat = parser[1].split('.')
smoothing_radius = int(parser[2])
src = cv2.VideoCapture("{}.{}".format(vidname, vidformat))

ret, prev = src.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
prev_features = cv2.goodFeaturesToTrack(prev_gray, 200, 0.01, 30)
last_transform, rough_transform, rough_trajectory, smooth_trajectory, smooth_transform = [], [], [], [], []

a, x, y = 0, 0, 0
rough_trajectory_out = open("rough_traj.txt", "w")
rough_transform_out = open("rough_trans.txt", "w")
while src.isOpened():
    ret, cur = src.read()
    if not ret:
        break
    cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    cur_features = cv2.goodFeaturesToTrack(cur_gray, 200, 0.01, 30)
    cur_features, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_features, cur_features)
    prev_tracked_features = prev_features[np.where(status == 1)[0]]
    cur_tracked_features = cur_features[np.where(status == 1)[0]]
    trans, ret2 = cv2.estimateAffinePartial2D(prev_tracked_features, cur_tracked_features)
    if trans is None:
        trans = last_transform
    last_transform = trans
    dx, dy, da = trans[0][2], trans[1][2], np.arctan2(trans[1][0], trans[0][0])
    x, y, a = x+dx, y+dy, a+da
    rough_trajectory.append([x, y, a])
    rough_transform.append([dx, dy, da])
    prev, prev_gray, prev_features = cur, cur_gray, cur_features
    rough_trajectory_out.write("{};{};{}\n".format(x, y, a))
    rough_transform_out.write("{};{};{}\n".format(dx, dy, da))
rough_transform_out.close()
rough_trajectory_out.close()

smooth_trajectory_out = open("smooth_traj.txt", "w")
for traj in range(len(rough_trajectory)):
    traj_sum, k = [0, 0, 0], 0
    for dist in range(-smoothing_radius, smoothing_radius):
        if 0 <= traj+dist < len(rough_trajectory):
            traj_sum = [x + y for x, y in zip(traj_sum, rough_trajectory[traj+dist])]
            k += 1
    avg = [x/k for x in traj_sum]
    smooth_trajectory.append(avg)
    smooth_trajectory_out.write("{};{};{}\n".format(avg[0], avg[1], avg[2]))
smooth_trajectory_out.close()

smooth_transform_out = open("smooth_trans.txt", "w")
trans_sum = [0, 0, 0]
for trans in range(len(rough_transform)):
    #trans_d = [x + y - z for x, y, z in zip(rough_transform[trans], smooth_trajectory[trans], rough_trajectory[trans])]
    trans_d = [x - y for x, y in zip(smooth_trajectory[trans], rough_trajectory[trans])]
    smooth_transform.append(trans_d)
    smooth_transform_out.write("{};{};{}\n".format(trans_d[0], trans_d[1], trans_d[2]))
smooth_transform_out.close()

src.set(cv2.CAP_PROP_POS_FRAMES, 0)
horiz, vert = src.get(cv2.CAP_PROP_FRAME_WIDTH), src.get(cv2.CAP_PROP_FRAME_HEIGHT)
diag = int(math.sqrt(vert*vert+horiz*horiz))
out = cv2.VideoWriter("{}_stab{}.{}".format(vidname, smoothing_radius, vidformat), cv2.VideoWriter_fourcc(*'XVID'), src.get(cv2.CAP_PROP_FPS), (int(horiz), int(vert)))
rotonly = 0
for trans in tqdm(smooth_trajectory):
    ret, cur = src.read()
    if not ret:
        break
    cur = cv2.warpAffine(cur, np.array([[math.cos(trans[2]), -math.sin(trans[2]), trans[0]*(1-rotonly)], [math.sin(trans[2]), math.cos(trans[2]), trans[1]*(1-rotonly)]]), (int(horiz), int(vert)))
    out.write(cur)
out.release()
src.release()

fig, axes = plt.subplots(2, 2, True)
axes[0, 0].set_title("rough transformation")
axes[0, 0].plot(rough_transform)
axes[0, 1].set_title("rough trajectory")
axes[0, 1].plot(rough_trajectory)
axes[1, 0].set_title("smooth transformation")
axes[1, 0].plot(smooth_transform)
axes[1, 1].set_title("smooth trajectory")
axes[1, 1].plot(smooth_trajectory)
plt.show()
