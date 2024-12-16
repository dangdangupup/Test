"""
    测试了：利用数学表达
"""
# =====================================
import os.path as osp
import cv2
import numpy as np
import time


def draw_kps(img, kps, color=(0,0, 255), r = 3):
    for kp in kps:
        img = cv2.circle(img, kp.astype(int), r, color, thickness=2)
    return img

def draw_lines(img, kps_1, kps_2, c=(255, 0, 0)):
    for kp1, kp2 in zip(kps_1, kps_2):
        img = cv2.line(img, kp1.astype(int), kp2.astype(int), color=c, thickness=2)
    return img


img_dir = '/Users/ed/Data/Image/optical_flow/1'
img_name_1 = '1.jpg'
img_name_2 = '2.jpg'

img_path_1 = osp.join(img_dir, img_name_1)
img_path_2 = osp.join(img_dir, img_name_2)

img_1 = cv2.imread(img_path_1)
img_2 = cv2.imread(img_path_2)

img_g_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img_g_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

kps = cv2.goodFeaturesToTrack(img_g_1, 100, 0.01, 10, None)
kps = kps.reshape(-1, 2)

next_kps = None
# --- Pyramid LK --- 
# next_kps, stat, err = cv2.calcOpticalFlowPyrLK(img_g_1, img_g_2, kps, None, winSize=(50, 50), maxLevel=3);     
# img_1 = draw_kps(img_1, kps, r=3)
# img_1 = draw_kps(img_1, next_kps, (0, 255, 0), r=5)
# img_1 = draw_lines(img_1, kps, next_kps)
# cv2.imshow("optical", img_1)

# # --- DISOpticalFlow --- 
st = time.time()
alg = cv2.DISOpticalFlow().create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
alg_kps = alg.calc(img_g_1, img_g_2, None)
et = time.time()
print(f'Cost time: {et - st}')

flow_img = np.linalg.norm(alg_kps, axis=-1)
flow_img /= flow_img.max()
flow_img = (1.0 - flow_img[:, :, np.newaxis]) * np.array([1.0, 0, 0]) + flow_img[:, :, np.newaxis] * np.array([0, 0, 1.0])
cv2.imshow("flow", flow_img)


cv2.waitKey(-1)
cv2.destroyAllWindows()

optical = cv2.DISOpticalFlow().create()

# ====== ref code ======
# core: 
#   - video process
#   - 
# ====== end ======

# import numpy as np 
# import cv2 

# cap = cv2.VideoCapture('sample.mp4') 

# # params for corner detection 
# feature_params = dict( maxCorners = 100, 
# 					qualityLevel = 0.3, 
# 					minDistance = 7, 
# 					blockSize = 7 ) 

# # Parameters for lucas kanade optical flow 
# lk_params = dict( winSize = (15, 15), 
# 				maxLevel = 2, 
# 				criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
# 							10, 0.03)) 

# # Create some random colors 
# color = np.random.randint(0, 255, (100, 3)) 

# # Take first frame and find corners in it 
# ret, old_frame = cap.read() 
# old_gray = cv2.cvtColor(old_frame, 
# 						cv2.COLOR_BGR2GRAY) 
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, 
# 							**feature_params) 

# # Create a mask image for drawing purposes 
# mask = np.zeros_like(old_frame) 

# while(1): 
	
# 	ret, frame = cap.read() 
# 	frame_gray = cv2.cvtColor(frame, 
# 							cv2.COLOR_BGR2GRAY) 

# 	# calculate optical flow 
# 	p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, 
# 										frame_gray, 
# 										p0, None, 
# 										**lk_params) 

# 	# Select good points 
# 	good_new = p1[st == 1] 
# 	good_old = p0[st == 1] 

# 	# draw the tracks 
# 	for i, (new, old) in enumerate(zip(good_new, 
# 									good_old)): 
# 		a, b = new.ravel() 
# 		c, d = old.ravel() 
# 		mask = cv2.line(mask, (a, b), (c, d), 
# 						color[i].tolist(), 2) 
		
# 		frame = cv2.circle(frame, (a, b), 5, 
# 						color[i].tolist(), -1) 
		
# 	img = cv2.add(frame, mask) 

# 	cv2.imshow('frame', img) 
	
# 	k = cv2.waitKey(25) 
# 	if k == 27: 
# 		break

# 	# Updating Previous frame and points 
# 	old_gray = frame_gray.copy() 
# 	p0 = good_new.reshape(-1, 1, 2) 

# cv2.destroyAllWindows() 
# cap.release() 
