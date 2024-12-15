"""
    测试了：分形中的 Julia 表达式
"""

import cv2
import numpy as np

lr = 2
x = np.linspace(-lr, lr, 256).reshape(1, -1)
y = np.linspace(-lr, lr, 256).reshape(-1, 1)
org_img = x + y*1j

def julia(data, c=0.75):
    return data**2 - c

def julia(data, c=0.75):
    data = data**2 - c
    b_over = np.abs(data) > 100
    data[b_over] = 0.0
    return data, b_over


def generate_img(c=0.75):
    global org_img
    img = np.copy(org_img)
    b_over = np.zeros(img.shape).astype(bool)
    for i in range(20):

        img, bo = julia(img, c+0.1*c*1j)
        b_over = b_over | bo

    img = np.abs(img)
    img = img * 1 / np.max(img)
    img[b_over] = 1
    # print(np.isinf(img).sum())
    # print(np.max(img), np.min(img), np.mean(img), np.var(img))
    return img

# ref_c = 0.8+0.15j
# ref_c = -1.755
ref_c = 0.123 + 0.745j
img = generate_img(ref_c)
cv2.imshow("Julia", img)
# cv2.waitKey(-1)
# exit(-1)
t = 0
last_c = t
is_pause = False
while True:
    key = cv2.waitKey(100)
    if key == ord('q'):
        exit(0)
    elif key == ord(' '):
        is_pause = ~is_pause
        
    if is_pause:
        c = last_c
    else:
        # c = np.sin(t + 0.75)
        c = np.sin(t) + ref_c


    img = generate_img(c = c)
    cv2.putText(img, f'{c:.3f}', [24, 24],cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0])
    cv2.imshow("Julia", img)
    t += 0.1
    last_c = c


