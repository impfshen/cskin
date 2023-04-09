#_*_coding:utf8_*_
# 将图像按照标准色块调整到标准色
import os
import cv2
import numpy as np
from seetaUtil import SampleWapper
from detect_color import get_color_horizontal, get_color_vertical
from filepath import colorPath

def findStandardColorArea(whiteimg,y1,y2,x1,x2,name,direction):
    height, width = whiteimg.shape[:2]
    if height > width:
        standard_color_pre_area = whiteimg[y1:y2, x1:x2]
        blue, green, red = get_color_vertical(standard_color_pre_area,name,direction)
    else:
        blue, green, red = get_color_horizontal(whiteimg,name,direction)
    return blue, green, red


def calcStandardAreaColorParmeter(color_area) :

    bgr_b = color_area[:, :, 0]
    average_bgr_b = cv2.mean(bgr_b)
    bgr_g = color_area[:, :, 1]
    average_bgr_g = cv2.mean(bgr_g)
    bgr_r = color_area[:, :, 2]
    average_bgr_r = cv2.mean(bgr_r)
    lab = cv2.cvtColor(color_area, cv2.COLOR_BGR2LAB)
    lab_l = lab[:, :, 0]
    average_l = cv2.mean(lab_l)
    lab_a = lab[:, :, 1]
    average_a = cv2.mean(lab_a)
    lab_b = lab[:, :, 2]
    average_b = cv2.mean(lab_b)
    return average_bgr_b,average_bgr_g,average_bgr_r,average_l,average_a,average_b



def computeRefAreaColor(whiteimg,name):
    #截取标准色块所在的位置 5036,5096,1411,2000(5184*3456)
    #截取标准色块所在的位置 735,740,214,306(800*537)
    #findStandardColorArea(whiteimg,735,740,214,306,name,num)
    #name = img_middle_white
    direction = name.split('_',2)[1]
    color = name.split('_',2)[2]
    if direction == 'left':
        blue, green, red = findStandardColorArea(whiteimg,4665,5096,1728,3456,name,direction)
    elif direction == 'right':
        blue, green, red = findStandardColorArea(whiteimg,4665,5096,0,1728,name,direction)
    elif direction == 'middle':
        blue, green, red = findStandardColorArea(whiteimg,4665,5096,864,2592,name,direction)
    else:
        print('invalid filename')
    return True, blue, green, red



if __name__ == "__main__":
    print(None)

