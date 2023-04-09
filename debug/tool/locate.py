#_*_coding:utf8_*_
import os
import cv2
import numpy as np
import sys
sys.path.append(r'/home/cskin')
from tool.filepath import recordPath
from src_terminal.filepath import locationPath


def locateMiddleBlueArea(src):
    thresh1 = np.array([77, 50, 45])
    thresh2 = np.array([97, 60, 55])
    blue = cv2.inRange(src, thresh1, thresh2)
    #cv2.imwrite('/home/spf/other/findCenterPoint/blue.jpg',blue)
    cv2.erode(blue, (3,3), blue, iterations= 3)
    cv2.dilate(blue, (3,3), blue, iterations= 3)
    cnts, _ = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts_sort = sorted(cnts, key= cv2.contourArea, reverse= True)
    box = cv2.minAreaRect(cnts_sort[0])
    points = np.int0(cv2.cv.BoxPoints(box))
    y = (points[0,1] + points[2,1]) // 2
    x = (points[0,0] + points[2,0]) // 2
    return y, x


def locateMiddleGreenArea(src):
    thresh1 = np.array([60, 155, 85])
    thresh2 = np.array([85, 175, 95])
    blue = cv2.inRange(src, thresh1, thresh2)
    #cv2.imwrite('/home/spf/other/findCenterPoint/green.jpg',blue)
    cv2.erode(blue, (3,3), blue, iterations= 3)
    cv2.dilate(blue, (3,3), blue, iterations= 3)
    cnts, _ = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts_sort = sorted(cnts, key= cv2.contourArea, reverse= True)
    box = cv2.minAreaRect(cnts_sort[0])
    points = np.int0(cv2.cv.BoxPoints(box))
    y = (points[0,1] + points[2,1]) // 2
    x = (points[0,0] + points[2,0]) // 2
    return y, x


def locateMiddleRedArea(src):
    thresh1 = np.array([35, 45, 142])
    thresh2 = np.array([45, 55, 162])
    blue = cv2.inRange(src, thresh1, thresh2)
    #cv2.imwrite('/home/spf/other/findCenterPoint/red.jpg',blue)
    cv2.erode(blue, (3,3), blue, iterations= 3)
    cv2.dilate(blue, (3,3), blue, iterations= 3)
    cnts, _ = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts_sort = sorted(cnts, key= cv2.contourArea, reverse= True)
    box = cv2.minAreaRect(cnts_sort[0])
    points = np.int0(cv2.cv.BoxPoints(box))
    y = (points[0,1] + points[2,1]) // 2
    x = (points[0,0] + points[2,0]) // 2
    return y, x


def locateLeftBlueArea(src):
    thresh1 = np.array([50, 30, 25])
    thresh2 = np.array([70, 40, 35])
    blue = cv2.inRange(src, thresh1, thresh2)
    #cv2.imwrite('/home/spf/other/findCenterPoint/blue.jpg',blue)
    cv2.erode(blue, (3,3), blue, iterations= 3)
    cv2.dilate(blue, (3,3), blue, iterations= 3)
    cnts, _ = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts_sort = sorted(cnts, key= cv2.contourArea, reverse= True)
    box = cv2.minAreaRect(cnts_sort[0])
    points = np.int0(cv2.cv.BoxPoints(box))
    y = (points[0,1] + points[2,1]) // 2
    x = (points[0,0] + points[2,0]) // 2
    return y, x


def locateLeftGreenArea(src):
    thresh1 = np.array([47, 130, 62])
    thresh2 = np.array([57, 150, 72])
    blue = cv2.inRange(src, thresh1, thresh2)
    #cv2.imwrite('/home/spf/other/findCenterPoint/green.jpg',blue)
    cv2.erode(blue, (3,3), blue, iterations= 3)
    cv2.dilate(blue, (3,3), blue, iterations= 3)
    cnts, _ = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts_sort = sorted(cnts, key= cv2.contourArea, reverse= True)
    box = cv2.minAreaRect(cnts_sort[0])
    points = np.int0(cv2.cv.BoxPoints(box))
    y = (points[0,1] + points[2,1]) // 2
    x = (points[0,0] + points[2,0]) // 2
    return y, x


def locateLeftRedArea(src):
    thresh1 = np.array([15, 25, 102])
    thresh2 = np.array([25, 35, 132])
    blue = cv2.inRange(src, thresh1, thresh2)
    #cv2.imwrite('/home/spf/other/findCenterPoint/red.jpg',blue)
    cv2.erode(blue, (3,3), blue, iterations= 3)
    cv2.dilate(blue, (3,3), blue, iterations= 3)
    cnts, _ = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts_sort = sorted(cnts, key= cv2.contourArea, reverse= True)
    box = cv2.minAreaRect(cnts_sort[0])
    points = np.int0(cv2.cv.BoxPoints(box))
    y = (points[0,1] + points[2,1]) // 2
    x = (points[0,0] + points[2,0]) // 2
    return y, x


def locateRightBlueArea(src):
    thresh1 = np.array([60, 30, 25])
    thresh2 = np.array([80, 40, 35])
    blue = cv2.inRange(src, thresh1, thresh2)
    #cv2.imwrite('/home/spf/other/findCenterPoint/blue.jpg',blue)
    cv2.erode(blue, (3,3), blue, iterations= 3)
    cv2.dilate(blue, (3,3), blue, iterations= 3)
    cnts, _ = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts_sort = sorted(cnts, key= cv2.contourArea, reverse= True)
    box = cv2.minAreaRect(cnts_sort[0])
    points = np.int0(cv2.cv.BoxPoints(box))
    y = (points[0,1] + points[2,1]) // 2
    x = (points[0,0] + points[2,0]) // 2
    return y, x


def locateRightGreenArea(src):
    thresh1 = np.array([35, 130, 55])
    thresh2 = np.array([55, 150, 65])
    blue = cv2.inRange(src, thresh1, thresh2)
    #cv2.imwrite('/home/spf/other/findCenterPoint/green.jpg',blue)
    cv2.erode(blue, (3,3), blue, iterations= 3)
    cv2.dilate(blue, (3,3), blue, iterations= 3)
    cnts, _ = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts_sort = sorted(cnts, key= cv2.contourArea, reverse= True)
    box = cv2.minAreaRect(cnts_sort[0])
    points = np.int0(cv2.cv.BoxPoints(box))
    y = (points[0,1] + points[2,1]) // 2
    x = (points[0,0] + points[2,0]) // 2
    return y, x


def locateRightRedArea(src):
    thresh1 = np.array([20, 20, 123])
    thresh2 = np.array([35, 40, 143])
    blue = cv2.inRange(src, thresh1, thresh2)
    #cv2.imwrite('/home/spf/other/findCenterPoint/red.jpg',blue)
    cv2.erode(blue, (3,3), blue, iterations= 3)
    cv2.dilate(blue, (3,3), blue, iterations= 3)
    cnts, _ = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts_sort = sorted(cnts, key= cv2.contourArea, reverse= True)
    box = cv2.minAreaRect(cnts_sort[0])
    points = np.int0(cv2.cv.BoxPoints(box))
    y = (points[0,1] + points[2,1]) // 2
    x = (points[0,0] + points[2,0]) // 2
    return y, x


def locateMiddlePoint(path):
    img = cv2.imread(path)
    #print(img[1765][4710])
    #print(img[1553][4696])
    #print(img[1988][4712])
    src = img[1350:2180, 4500:5100]
    blue_y, blue_x = locateMiddleBlueArea(src)
    green_y, green_x = locateMiddleGreenArea(src)
    red_y, red_x = locateMiddleRedArea(src)
    return blue_y, blue_x, green_y, green_x, red_y, red_x
    
    
def locateLeftPoint(path):
    img = cv2.imread(path)
    #print(img[665][245])
    #print(img[828][207])
    #print(img[512][290])
    src = img[425:1050, 0:420]
    blue_y, blue_x = locateLeftBlueArea(src)
    green_y, green_x = locateLeftGreenArea(src)
    red_y, red_x = locateLeftRedArea(src)
    return blue_y, blue_x, green_y, green_x, red_y, red_x


def locateRightPoint(path):
    img = cv2.imread(path)
    #print(img[741][4957])
    #print(img[594][4892])
    #print(img[904][5006])
    src = img[480:1100, 4800:5180]
    blue_y, blue_x = locateRightBlueArea(src)
    green_y, green_x = locateRightGreenArea(src)
    red_y, red_x = locateRightRedArea(src)
    return blue_y, blue_x, green_y, green_x, red_y, red_x


def findCenterPoint(path, locationPath):
    filelist = os.listdir(path)
    for filename in filelist:
        print(filename)
        if filename == 'img_middle_white.jpg':
            blue_y, blue_x, green_y, green_x, red_y, red_x = locateMiddlePoint(path+'/'+filename)
            f1=open(locationPath+'/ColorPointMiddle.txt', 'w+')
            f1.write(str(float(blue_y+1350))+'\n'+str(float(blue_x+4500))+'\n'+str(float(green_y+1350))+'\n'+str(float(green_x+4500))+'\n'+str(float(red_y+1350))+'\n'+str(float(red_x+4500)))
            f1.close()
        if filename == 'img_left_white.jpg':
            blue_y, blue_x, green_y, green_x, red_y, red_x = locateLeftPoint(path+'/'+filename)
            f2=open(locationPath+'/ColorPointLeft.txt', 'w+')
            f2.write(str(float(blue_y+425))+'\n'+str(float(blue_x))+'\n'+str(float(green_y+425))+'\n'+str(float(green_x))+'\n'+str(float(red_y+425))+'\n'+str(float(red_x)))
            f2.close()
        if filename == 'img_right_white.jpg':
            blue_y, blue_x, green_y, green_x, red_y, red_x = locateRightPoint(path+'/'+filename)
            f3=open(locationPath+'/ColorPointRight.txt', 'w+')
            f3.write(str(float(blue_y+480))+'\n'+str(float(blue_x+4800))+'\n'+str(float(green_y+480))+'\n'+str(float(green_x+4800))+'\n'+str(float(red_y+480))+'\n'+str(float(red_x+4800)))
            f3.close()


if __name__ == "__main__":
    findCenterPoint(recordPath,locationPath)
