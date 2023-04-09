# -*- coding: utf-8 -*-
import cv2
#import imutils
import numpy as np

def enhance_uv(img):
    level_img = img
    #level_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
    for i in range(1000,2450):
        for j in range(2000,3300):
            if 84 < img[i, j, 2] < 128 and img[i, j, 0] < 192 and img[i, j, 1] < 192:
                #level_img[i, j, 2] = max(int(img[i, j, 2] * 12), 255)
                #level_img[i, j, 1] = min(int(img[i, j, 1] // 10)+59, 100)
                #level_img[i, j, 0] = 0
                level_img[i, j] = [0, 69, 255]
            elif img[i, j, 1] < 100 and img[i, j, 1] > 80 and img[i, j, 2] > 80:
                #level_img[i, j, 2] = max(int(img[i, j, 2] // 5)+155, 255)
                #level_img[i, j, 1] = max(int(img[i, j, 1] // 5)+155, 255)
                #level_img[i, j, 0] = max(int(img[i, j, 0] // 5)+155, 255)
                level_img[i, j] = [255, 255, 255]
            else:
                level_img[i, j] = img[i, j]
    return level_img
