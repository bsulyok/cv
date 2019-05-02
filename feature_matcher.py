import cv2
import sys


parser = sys.argv
src = cv2.VideoCapture(parser[1])
out_name = (parser[1].split('.'))[0]+'_feat.'+(parser[1].split('.'))[1]
out_dim = (int(src.get(cv2.CAP_PROP_FRAME_WIDTH)), int(src.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'XVID'), int(src.get(cv2.CAP_PROP_FPS)), out_dim)
orb = cv2.ORB_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

ret, prev = src.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
kp_prev, des_prev = orb.detectAndCompute(prev_gray, None)

while(src.isOpened()):
    ret, cur = src.read()
    if not ret:
        break
    frame = cur.copy()
    cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    kp_cur, des_cur = orb.detectAndCompute(cur, None)

    matches = matcher.match(des_prev, des_cur)
    good = []
    for m in matches:
        if m.distance < 0.75:
            good.append(m)
    #good = [m for (m,n) in matches if m.distance < 0.75*n.distance]
    
    frame = cv2.drawMatches(prev_gray, kp_prev, cur_gray, kp_cur, good, frame)

    out.write(frame)
    
    prev_gray, kp_prev, des_prev = cur_gray, kp_cur, des_cur
src.release()
out.release()
cv2.destroyAllWindows() 
    
