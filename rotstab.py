import cv2
import math
import numpy as np
import sys
import cv3

parser = sys.argv
vidname, vidformat = parser[1].split('.')
smoothing_radius = int(parser[2])
src = cv2.VideoCapture("{}.{}".format(vidname,vidformat))

ret, prev = src.read()
prev_gray = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
prev_corner = cv2.goodFeaturesToTrack(prev_gray,200,0.01,30)
last_transform, prev_to_cur_transform, trajectory, smooth_trajectory, smooth_prev_to_cur_transform = [], [], [], [], []
a = 0
output_traj=open("output_traj.txt",'w')
while(src.isOpened()):
    ret, cur = src.read()
    if not ret:
        break
    cur_gray = cv2.cvtColor(cur,cv2.COLOR_BGR2GRAY)
    cur_corner = cv2.goodFeaturesToTrack(cur_gray,200,0.01,30)
    cur_corner, status, err = cv2.calcOpticalFlowPyrLK(prev_gray,cur_gray,prev_corner,cur_corner)
    prev_corner2 = prev_corner[np.where(status==1)[0]]
    cur_corner2 = cur_corner[np.where(status==1)[0]]
    transform, ret2 = cv2.estimateAffinePartial2D(prev_corner2,cur_corner2)
    if transform is None:
        transform = last_transform
    last_transform = transform
    da = np.arctan2(transform[1][0],transform[0][0])*180/math.pi
    a += da
    trajectory.append(a)
    prev_to_cur_transform.append(da)
    output_traj.write("{}\n".format(da))
    prev, prev_gray, prev_corner = cur, cur_gray, cur_corner
output_traj.close()


output_smoothtraj = open("output_smoothtraj.txt",'w')
for traj in range(len(trajectory)):
    sum_a, k = 0, 0
    for dist in range(-smoothing_radius,smoothing_radius):
        if 0 <= traj+dist < len(trajectory):
            sum_a += trajectory[traj+dist]
            k += 1
    smooth_trajectory.append(sum_a/k)
    output_smoothtraj.write("{}\n".format(sum_a/k))
output_smoothtraj.close()

output_trans=open("output_smoothtrans.txt",'w')
a = 0
for trans in range(len(prev_to_cur_transform)):
    a += prev_to_cur_transform[trans]
    smooth_prev_to_cur_transform.append(prev_to_cur_transform[trans] + smooth_trajectory[trans] - a)
    output_trans.write("{}\n".format(prev_to_cur_transform[trans] + smooth_trajectory[trans] - a))
output_trans.close()

src.set(cv2.CAP_PROP_POS_FRAMES, 0)
horiz, vert = src.get(cv2.CAP_PROP_FRAME_WIDTH), src.get(cv2.CAP_PROP_FRAME_HEIGHT)
diag = int(math.sqrt(vert*vert+horiz*horiz))
out = cv2.VideoWriter("{}_rotstab{}.{}".format(vidname,smoothing_radius,vidformat) , cv2.VideoWriter_fourcc(*'XVID'), src.get(cv2.CAP_PROP_FPS), (diag,diag))
k = 0

for angle in smooth_prev_to_cur_transform:
    ret, cur = src.read()
    if not ret:
        break
    cur = cv3.pad_rotate(cur,angle)
    out.write(cur) 

src.release()
out.release()
cv2.destroyAllWindows() 


