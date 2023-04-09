import os
import cv2
import numpy as np
import colorList
import math
from skimage import exposure
import matplotlib.pyplot as plt
from PIL import Image, ImageStat
import sys
sys.path.append(r'/home/cskin')
from tool.filepath import standardImg, colorPath
from src_terminal.filepath import standardPath
from rotate import direction_correct


def image_brightness(rgb_image):
    stat = ImageStat.Stat(rgb_image)
    r, g, b = stat.rms
    return math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))


def findStandardColorArea(img,filename):
    color = filename.rstrip('.jpg')
    if not os.access(colorPath+'/{}'.format(color),os.F_OK):
        os.mkdir(colorPath+'/{}'.format(color))
    if img.shape[0] > img.shape[1]:
        red = img[5058:5063,1468:1473]
        blue = img[5043:5048,1693:1698]
        green = img[4998:5003,1948:1953]
    else:
        red = img[1984:1989,5058:5063]
        blue = img[1759:1764,5043:5048]
        green = img[1504:1507,4998:5003]
    cv2.imwrite(colorPath+'/'+color+'/'+'red.jpg',red)
    cv2.imwrite(colorPath+'/'+color+'/'+'blue.jpg',blue)
    cv2.imwrite(colorPath+'/'+color+'/'+'green.jpg',green)


def computeRefAreaColor(path,filename):
    img = cv2.imread(path+'/'+filename)
    findStandardColorArea(img,filename)
    color = filename.rstrip('.jpg')
    if not os.access(colorPath+'/{}/blue.jpg'.format(color),os.F_OK):
        return False, None, None, None
    if not os.access(colorPath+'/{}/green.jpg'.format(color),os.F_OK):
        return False, None, None, None
    if not os.access(colorPath+'/{}/red.jpg'.format(color),os.F_OK):
        return False, None, None, None
    standard_area_b = cv2.imread(colorPath+'/{}/blue.jpg'.format(color))
    standard_area_g = cv2.imread(colorPath+'/{}/green.jpg'.format(color))
    standard_area_r = cv2.imread(colorPath+'/{}/red.jpg'.format(color))
    return True, standard_area_b, standard_area_g, standard_area_r


# path - address of one standard image
# filename - full name of standard image
def StandardColor(path, filename):
    flag, standard_area_b, standard_area_g, standard_area_r = computeRefAreaColor(path, filename)
    if not flag:
        print('no standard color card')
        return
    standard_area_b = Image.fromarray(cv2.cvtColor(standard_area_b, cv2.COLOR_BGR2RGB))
    standard_area_g = Image.fromarray(cv2.cvtColor(standard_area_g, cv2.COLOR_BGR2RGB))
    standard_area_r = Image.fromarray(cv2.cvtColor(standard_area_r, cv2.COLOR_BGR2RGB))
    brightness_b = image_brightness(standard_area_b)
    brightness_g = image_brightness(standard_area_g)
    brightness_r = image_brightness(standard_area_r)
    color=filename.rstrip('.jpg')
    f=open(standardPath+'/{}_standard.txt'.format(color), 'w')
    f.write(str(brightness_b)+'\n')
    f.write(str(brightness_g)+'\n')
    f.write(str(brightness_r)+'\n')


if __name__ == "__main__":
    StandardColor(standardImg, 'blue.jpg')
    StandardColor(standardImg, 'green.jpg')
    StandardColor(standardImg, 'red.jpg')
    StandardColor(standardImg, 'white.jpg')
    StandardColor(standardImg, 'upw.jpg')
    StandardColor(standardImg, 'uv.jpg')
    StandardColor(standardImg, 'brown.jpg')
