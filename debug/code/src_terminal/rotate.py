import cv2
import numpy as np
import os

def detection(path):

    img_src = cv2.imread(path)
    height, width = img_src.shape[:2]

    if height < width:
        img_gray = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        left=0
        right=0
        for i in range(height):
            left += int(img_gray[i][0])
            right += int(img_gray[i][width-1])
        brightness_left = left // height
        brightness_right = right // height
        
        if brightness_left > brightness_right:
            img = rotation(img_src,height,width)
            height, width = img.shape[:2]
            img = horizental(img,height,width)
        else:
            img = rotation(img_src,height,width)
    
    else:
        img = img_src
    
    return img


def rotation(img_src,height,width):

    map_x = np.zeros([width, height], np.float32)
    map_y = np.zeros([width, height], np.float32)

    for i in range(width):
        for j in range(height):
            map_x.itemset((i, j), i)
            map_y.itemset((i, j), height-1-j)

    img_dst = cv2.remap(img_src, map_x, map_y, cv2.INTER_LINEAR)

    return img_dst


def horizental(img_src,height,width):

    map_x = np.zeros([height, width], np.float32)
    map_y = np.zeros([height, width], np.float32)

    for i in range(height):
        for j in range(width):
            map_x.itemset((i, j), width-1-j)
            map_y.itemset((i, j), height-1-i)

    img_dst = cv2.remap(img_src, map_x, map_y, cv2.INTER_LINEAR)

    return img_dst


def overside(img_src,height,width):

    map_x = np.zeros([height, width], np.float32)
    map_y = np.zeros([height, width], np.float32)

    for i in range(height):
        for j in range(width):
            map_x.itemset((i, j), j)
            map_y.itemset((i, j), height-1-i)

    img_dst = cv2.remap(img_src, map_x, map_y, cv2.INTER_LINEAR)

    return img_dst


def direction_correct(path):
    img = detection(path)
    cv2.imwrite(path,img)


if __name__ == '__main__':
    direction_correct('/home/spf/other/test.jpg')
