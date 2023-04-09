#_*_coding:utf8_*_

import cv2
import numpy as np
from filepath import locationPath

def adjust_color_test(filename,filePath,resultPath):
    FileName=filename.rstrip('.jpg')
    direction=FileName.split('_',2)[1]

    if direction == 'left':
        LocationData=open(locationPath+'/ColorPointLeft.txt')
    elif direction == 'right':
        LocationData=open(locationPath+'/ColorPointRight.txt')
    elif direction == 'middle':
        LocationData=open(locationPath+'/ColorPointMiddle.txt')
    else:
        print(filename+': invalid filename')
        return

    point=[]
    for line in LocationData:
        point.append(int(float(line[:-1])))
    LocationData.close()

    img = cv2.imread(filePath, cv2.IMREAD_COLOR)
    cv2.circle(img, (point[1],point[0]), 7, (255, 255, 255), -1)
    cv2.circle(img, (point[3],point[2]), 7, (255, 255, 255), -1)
    cv2.circle(img, (point[5],point[4]), 7, (255, 255, 255), -1)
    cv2.imwrite(resultPath, img)
