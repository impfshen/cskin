#_*_coding:utf8_*_
import cv2
import numpy as np
from filepath import colorPath, locationPath


def get_color_horizontal(frame,name,direction):
    if direction == 'left':
        filedata = open(locationPath+'/'+'ColorPointLeft.txt')
        point=[0,0,0,0,0,0]
        i = 0
        for line in filedata:
           point[i] = int(float(line[:-1]))
           i = i + 1
        filedata.close()
        red = frame[(point[4]-2):(point[4]+3),(point[5]-2):(point[5]+3)]
        #cv2.imwrite(colorPath+'/'+name+'/red.jpg',red)
        blue = frame[(point[0]-2):(point[0]+3),(point[1]-2):(point[1]+3)]
        #cv2.imwrite(colorPath+'/'+name+'/blue.jpg',blue)
        green = frame[(point[2]-2):(point[2]+3),(point[3]-2):(point[3]+3)]
        #cv2.imwrite(colorPath+'/'+name+'/green.jpg',green)
        return blue, green, red
    elif direction == 'right':
        filedata = open(locationPath+'/'+'ColorPointRight.txt')
        point=[0,0,0,0,0,0]
        i = 0
        for line in filedata:
           point[i] = int(float(line[:-1]))
           i = i + 1
        filedata.close()
        red = frame[(point[4]-2):(point[4]+3),(point[5]-2):(point[5]+3)]
        #cv2.imwrite(colorPath+'/'+name+'/red.jpg',red)
        blue = frame[(point[0]-2):(point[0]+3),(point[1]-2):(point[1]+3)]
        #cv2.imwrite(colorPath+'/'+name+'/blue.jpg',blue)
        green = frame[(point[2]-2):(point[2]+3),(point[3]-2):(point[3]+3)]
        #cv2.imwrite(colorPath+'/'+name+'/green.jpg',green)
        return blue, green, red
    elif direction == 'middle':
        filedata = open(locationPath+'/'+'ColorPointMiddle.txt')
        point=[0,0,0,0,0,0]
        i = 0
        for line in filedata:
           point[i] = int(float(line[:-1]))
           i = i + 1
        filedata.close()
        red = frame[(point[4]-2):(point[4]+3),(point[5]-2):(point[5]+3)]
        #cv2.imwrite(colorPath+'/'+name+'/red.jpg',red)
        blue = frame[(point[0]-2):(point[0]+3),(point[1]-2):(point[1]+3)]
        #cv2.imwrite(colorPath+'/'+name+'/blue.jpg',blue)
        green = frame[(point[2]-2):(point[2]+3),(point[3]-2):(point[3]+3)]
        #cv2.imwrite(colorPath+'/'+name+'/green.jpg',green)
        return blue, green, red
    else:
        return None, None, None


def get_color_vertical(frame,name,direction):
    if direction == 'left':
        red = frame[250:255,1210:1215]
        #cv2.imwrite(colorPath+'/'+name+'/red.jpg',red)
        blue = frame[265:270,1050:1055]
        #cv2.imwrite(colorPath+'/'+name+'/blue.jpg',blue)
        green = frame[265:270,950:955]
        #cv2.imwrite(colorPath+'/'+name+'/green.jpg',green)
        return blue, green, red
    elif direction == 'right':
        green = frame[210:215,585:590]
        #cv2.imwrite(colorPath+'/'+name+'/green.jpg',green)
        blue = frame[240:245,725:730]
        #cv2.imwrite(colorPath+'/'+name+'/blue.jpg',blue)
        red = frame[275:280,830:835]
        #cv2.imwrite(colorPath+'/'+name+'/red.jpg',red)
        return blue, green, red
    elif direction == 'middle':
        green = frame[145:150,720:725]
        #cv2.imwrite(colorPath+'/'+name+'/green.jpg',green)
        blue = frame[145:150,875:880]
        #cv2.imwrite(colorPath+'/'+name+'/blue.jpg',blue)
        red = frame[145:150,1040:1045]
        #cv2.imwrite(colorPath+'/'+name+'/red.jpg',red)
        return blue, green, red
    else:
        return None, None, None


if __name__ == "__main__":
    '''
    img = cv2.imread(colorPath+'/frame.jpg',cv2.IMREAD_COLOR)
    y,x = get_color(img)
    print(y,x)
    '''
