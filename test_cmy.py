"""
    测试内容
"""
import cv2
import numpy as np


def cmy(image):
    data = 255-image
    return data
    # data = min(data, axis=-1)



path = '/Users/ed/Pictures/v2-e9e5b99e7a5daaf838dbd6ec8d99ceae_1440w.webp'
img = cv2.imread(path)
cv2.imshow('test_1', img)


max_v = np.max(img, axis=-1, keepdims=True)
min_v = np.min(img, axis=-1, keepdims=True)
# img = max_v + min_v - img
img = 255 - img

cv2.imshow('test_2', img)


cv2.waitKey(-1)