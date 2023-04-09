#_*_coding:utf8_*_

import math
import cv2
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
import shutil
import cv2 as cv
from PIL import Image, ImageStat
from adjustcolor import computeRefAreaColor
from filepath import standardPath, colorPath, testPath
from enhance import enhance_uv


def image_brightness1(rgb_image):
    w, h = rgb_image.size
    hsv_image = cv.cvtColor(np.array(rgb_image, 'f'), cv.COLOR_RGB2HSV)
    sum_brightness = np.sum(hsv_image[:, :, 2])
    area = w * h 
    avg = sum_brightness / area
    return avg


def image_brightness2(rgb_image):
    gray_image = rgb_image.convert('L')
    stat = ImageStat.Stat(gray_image)
    return stat.mean[0]


def image_brightness3(rgb_image):
    stat = ImageStat.Stat(rgb_image)
    r, g, b = stat.mean
    return math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))


def image_brightness4(rgb_image):
    stat = ImageStat.Stat(rgb_image)
    r, g, b = stat.rms
    return math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))


def calc_gamma(brightness, filename):
    filedata = open(testPath+'/{}.txt'.format(filename.rstrip('.jpg')))
    for line in filedata:
        k = float(line[:-1])
    filedata.close()
    print(k)
    return brightness / k


def array_to_image(image_arr):
    if len(image_arr.shape) == 3:
        r = Image.fromarray(np.uint8(image_arr[:, :, 0]))
        g = Image.fromarray(np.uint8(image_arr[:, :, 1]))
        b = Image.fromarray(np.uint8(image_arr[:, :, 2]))
        image = Image.merge("RGB", (r, g, b))
        return image
    elif len(image_arr.shape) == 2:
        return Image.fromarray(np.uint8(image_arr))


def image_gamma_transform(pil_im, gamma):
    image_arr = np.array(pil_im)
    image_arr2 = exposure.adjust_gamma(image_arr, gamma)
    return array_to_image(image_arr2)


def standard_data(filename):
    filename=filename.rstrip('.jpg')
    Str=filename.split('_',2)
    color=Str[2]
    filedata=color+'_standard.txt'
    StandardData=open(standardPath+'/'+filedata)
    s=[0.0,0.0,0.0]
    i=0
    for line in StandardData:
        s[i]=float(line[:-1])
        i=i+1
    StandardData.close()
    return s[0],s[1],s[2]


def ImgBlur(src):
    dst = cv2.GaussianBlur(src,(5,5),0,0)
    return dst


def AdjustColor2(filename,filePath,resultPath):
    
    if filename != 'img_middle_white.jpg' and filename != 'img_middle_upw.jpg' and filename != 'img_left_white.jpg' and filename != 'img_right_white.jpg':
        white = cv2.imread(filePath, cv2.IMREAD_COLOR)
        cv2.imwrite(resultPath, white)
        return
    
    if filename == 'img_middle_uv.jpg':
        white = cv2.imread(filePath, cv2.IMREAD_COLOR)
        cv2.imwrite(resultPath, white)
        return

    brightness_b2,brightness_g2,brightness_r2=standard_data(filename)

    if brightness_b2==0 and brightness_g2==0 and brightness_r2==0:
        originalImg=Image.open(filePath)
        originalImg.save(resultPath)
        print('{}: no standard color data'.format(filename))
        return

    white = cv2.imread(filePath, cv2.IMREAD_COLOR)
    
    flag, standard_area_b, standard_area_g, standard_area_r = computeRefAreaColor(white, filename.rstrip('.jpg'))
    
    if(not flag):
        originalImg=Image.open(filePath)
        originalImg.save(resultPath)
        print('{}: no color area'.format(filename))
        return
 
    standard_area_b = Image.fromarray(cv2.cvtColor(standard_area_b, cv2.COLOR_BGR2RGB))
    standard_area_g = Image.fromarray(cv2.cvtColor(standard_area_g, cv2.COLOR_BGR2RGB))
    standard_area_r = Image.fromarray(cv2.cvtColor(standard_area_r, cv2.COLOR_BGR2RGB))

    brightness_b = image_brightness4(standard_area_b)
    brightness_g = image_brightness4(standard_area_g)
    brightness_r = image_brightness4(standard_area_r)

    img = Image.open(filePath)

    result = brightness_b + brightness_g + brightness_r - (brightness_b2 + brightness_g2 + brightness_r2)

    result = result / 25
    
    if(image_brightness4(img)+result<0):
        img.save(resultPath)
        print('{}: invalid gamma'.format(filename))
    else:
        if filename == 'img_middle_uv.jpg':
            newimage = image_gamma_transform(img, 0.6)
        else:
            newimage = image_gamma_transform(img, calc_gamma(image_brightness4(img) + result, filename))
        img = cv2.cvtColor(np.asarray(newimage),cv2.COLOR_RGB2BGR)
        if filename == 'img_left_blue.jpg' or filename == 'img_middle_blue.jpg' or filename == 'img_right_blue.jpg' or filename == 'img_left_white.jpg' or filename == 'img_middle_white.jpg' or filename == 'img_right_white.jpg':
            resimg = ImgBlur(img)
            cv2.imwrite(resultPath, resimg)
        elif filename == 'img_middle_uv.jpg':
            white = cv2.imread(filePath, cv2.IMREAD_COLOR)
            cv2.imwrite(resultPath, white)
        else:
            cv2.imwrite(resultPath, img)
