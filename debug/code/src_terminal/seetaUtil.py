#_*_coding:utf8_*_
import numpy as np
import cv2
import os
import traceback
import math
from scipy import interpolate
import sys,os
import io
from PIL import Image
import subprocess

file_dir = os.path.dirname(__file__)
print ("Python Version {}".format(sys.version))
platform = sys.platform
if platform == "win32":
    pass
elif platform == "linux":
    import whatimage
    import pyheif

def drawPoints(img, result, radius=10, color=(0,0,255), thickness=8):
    '''
    把seetaFace检测结果画到人脸上显示,对整张脸的图和半张脸的图片都适用
    Input:
    img: 检测图片
    result：dict 检测结果

    以下点数不准

    关于脸部点的信息，结合assets中的drawPointsImg.jpg理解
    FaceProfile:脸的轮廓，有21个点，从左往右 [{'X':,'Y':},{},{}]
    LeftEye：左眼，有八个点，最左边开始，逆时针
    RightEye：右眼，有八个点，最右边开始，顺时针
    LeftEyeBrow：左眉毛，8个点，方向和LeftEye一样
    RightEyeBrow：右眉毛，8个点，方向和RigthEye一样
    Mouth：嘴，22个点，分外轮廓（12）和内轮廓（10）（嘴巴张开），方向和LeftEye 一样
    Nose:鼻子，13个点 第一个点是鼻头，方向和和LeftEye 一样，从最上方开始
    LeftPupil：1个点，左瞳孔
    RightPupil：1个点，右瞳孔
    '''
    FaceProfile=result['FaceShapeSet'][0]['FaceProfile']
    LeftEye=result['FaceShapeSet'][0]['LeftEye']
    RightEye=result['FaceShapeSet'][0]['RightEye']
    LeftEyeBrow=result['FaceShapeSet'][0]['LeftEyeBrow']
    RightEyeBrow=result['FaceShapeSet'][0]['RightEyeBrow']
    Mouth=result['FaceShapeSet'][0]['Mouth']
    Nose=result['FaceShapeSet'][0]['Nose']
    LeftPupil=result['FaceShapeSet'][0]['LeftPupil']
    RightPupil=result['FaceShapeSet'][0]['RightPupil']
    imgd=img.copy()
    for point in FaceProfile:
        cv2.circle(imgd,(point['X'],point['Y']),radius,color,thickness)#(w,h)
        # cv2.namedWindow("test1",2)
        # cv2.resizeWindow("test1",img.shape[0],img.shape[1])
        # cv2.imshow("test1",imgd)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    for point in LeftEye:
        img=cv2.circle(imgd,(point['X'],point['Y']),radius,color,thickness)
        # cv2.namedWindow("test2",2)
        # cv2.resizeWindow("test2",img.shape[0],img.shape[1])
        # cv2.imshow("test2",imgd)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    for point in RightEye:
        img=cv2.circle(imgd,(point['X'],point['Y']),radius,color,thickness)
        # cv2.namedWindow("test3",2)
        # cv2.resizeWindow("test3",img.shape[0],img.shape[1])
        # cv2.imshow("test3",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    for point in LeftEyeBrow:
        img=cv2.circle(imgd,(point['X'],point['Y']),radius,color,thickness)
        # cv2.namedWindow("test4",2)
        # cv2.resizeWindow("test4",img.shape[0],img.shape[1])
        # cv2.imshow("test4",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    for point in RightEyeBrow:
        img=cv2.circle(imgd,(point['X'],point['Y']),radius,color,thickness)
        # cv2.namedWindow("test5",2)
        # cv2.resizeWindow("test5",img.shape[0],img.shape[1])
        # cv2.imshow("test5",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    for point in Mouth:
        img=cv2.circle(imgd,(point['X'],point['Y']),radius,color,thickness)
        # cv2.namedWindow("test6",2)
        # cv2.resizeWindow("test6",img.shape[0],img.shape[1])
        # cv2.imshow("test6",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    for point in Nose:
        img=cv2.circle(imgd,(point['X'],point['Y']),radius,color,thickness)
        # cv2.namedWindow("test7",2)
        # cv2.resizeWindow("test7",img.shape[0],img.shape[1])
        # cv2.imshow("test7",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    for point in LeftPupil:
        img=cv2.circle(imgd,(point['X'],point['Y']),radius,color,thickness)
        # cv2.namedWindow("test8",2)
        # cv2.resizeWindow("test8",img.shape[0],img.shape[1])
        # cv2.imshow("test8",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    for point in RightPupil:
        img=cv2.circle(imgd,(point['X'],point['Y']),radius,color,thickness)
        # cv2.namedWindow("test9",2)
        # cv2.resizeWindow("test9",img.shape[0],img.shape[1])
        # cv2.imshow("test9",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    return imgd

def drawPointsArray(img,l,radius=10, color=(0,0,255), thickness=8):
    '''
    画点
    Input:
    img
    l:np.array [[x,y],[x,y]] or list
    '''
    img_temp=img.copy()
    if isinstance(l,np.ndarray):
        l=l.tolist()
    for point in l:
        cv2.circle(img_temp,(point[0],point[1]),radius,color,thickness)
    return img_temp

def findnaris(img,result):
    Nose=result['FaceShapeSet'][0]['Nose']
    s=50
    x2l=50
    y2u=130
    y2d=120
    Points=np.array([
    [Nose[2]['X']-x2l, Nose[2]['Y']],
    [Nose[4]['X']-x2l, Nose[4]['Y']],
    [Nose[6]['X']-x2l, Nose[6]['Y']],
    [Nose[4]['X'], Nose[4]['Y']],
    [Nose[8]['X'], Nose[8]['Y']],
    [Nose[1]['X'], Nose[1]['Y']],
    [Nose[9]['X'], Nose[9]['Y']],
    [Nose[7]['X']+x2l, Nose[7]['Y']],
    [Nose[5]['X']+x2l, Nose[5]['Y']],
    [Nose[3]['X']+x2l, Nose[3]['Y']]])

    h,w,_=img.shape
    mask = np.zeros((h,w), dtype='uint8')
    mask = cv2.fillPoly(mask, np.int32([Points]), (255, 255, 255))
    Indices=np.nonzero(mask)
    img_base = img.copy()
    img_base.fill(255)
    img_base[Indices]=img[Indices]
    img_baseG=cv2.cvtColor(img_base,cv2.COLOR_BGR2GRAY)
    _,img_binary=cv2.threshold(img_baseG, 90, 255, cv2.THRESH_BINARY)# 80
    img_binary=cv2.bitwise_not(img_binary,img_binary);
    # imgWindow("1",img_binary)
    contours, hierarchy = cv2.findContours(img_binary, mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
    img_temp=img.copy()
    PointsL=[]
    for c in contours:
        area = cv2.contourArea(c)

        if area>2500:
            print(area)
            c_temp=np.array(c).reshape(-1,2)
            PointsL.append(c_temp)
            img_temp = cv2.polylines(img_temp, [np.array(c_temp)],True,(0,0,255),3,lineType = cv2.LINE_8)
            # cv2.namedWindow("smallImg",2)
            # cv2.imshow("smallImg",img_temp)
            # cv2.waitKey(0)
    return PointsL

def findForeHand_skin(img,result):
    '''
    find forehand area through skin color
    Input:
        img: in color
        result : result of Facial features localization
    '''
    FaceProfile=result['FaceShapeSet'][0]['FaceProfile']
    LeftEyeBrow=result['FaceShapeSet'][0]['LeftEyeBrow']
    RightEyeBrow=result['FaceShapeSet'][0]['RightEyeBrow']
    Nose=result['FaceShapeSet'][0]['Nose']
    Mouth=result['FaceShapeSet'][0]['Mouth']
    LeftEye=result['FaceShapeSet'][0]['LeftEye']
    x2r=100
    y2d=150
    x2l=150
    y2u=0
    PointsR=np.array(
    [[Nose[2]['X']-x2r,Nose[2]['Y']],
    [Nose[4]['X']-x2r,Nose[4]['Y']],
    [Nose[6]['X']-x2r,Nose[6]['Y']],
    [Mouth[0]['X']-x2r,Mouth[0]['Y']],
    [FaceProfile[9]['X']+x2l,FaceProfile[9]['Y']-y2u],
    [FaceProfile[8]['X']+x2l,FaceProfile[8]['Y']-y2u],
    [FaceProfile[7]['X']+x2l,FaceProfile[7]['Y']],
    [FaceProfile[6]['X']+x2l,FaceProfile[6]['Y']],
    [FaceProfile[5]['X']+x2l,FaceProfile[5]['Y']],
    [FaceProfile[2]['X']+x2l,FaceProfile[2]['Y']],
    [LeftEye[0]['X'],LeftEye[0]['Y']+y2d],
    [LeftEye[5]['X'],LeftEye[5]['Y']+y2d],
    [LeftEye[3]['X'],LeftEye[3]['Y']+y2d],
    [LeftEye[7]['X'],LeftEye[7]['Y']+y2d],
    [LeftEye[1]['X'],LeftEye[1]['Y']+y2d]]
    )

    # img_temp=img.copy()
    # img_temp = cv2.polylines(img_temp, [np.array(PointsR)],True,(0,0,255),3,lineType = cv2.LINE_8)
    # cv2.namedWindow("smallImg",2)
    # cv2.imshow("smallImg",img_temp)
    # cv2.waitKey(0)
    # input()
    h,w,_=img.shape
    mask = np.zeros((h,w), dtype='uint8')
    mask = cv2.fillPoly(mask, np.int32([PointsR]), (255, 255, 255))
    Indices = np.where(mask!=0)
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    roi_hsv = img_hsv[Indices[0],Indices[1],:]
    h_mean = np.mean(roi_hsv[:,0])
    s_mean = np.mean(roi_hsv[:,1])
    v_mean = np.mean(roi_hsv[:,2])
    h_max = np.max(roi_hsv[:,0])
    h_min = np.min(roi_hsv[:,0])
    s_max = np.max(roi_hsv[:,1])
    s_min = np.min(roi_hsv[:,1])
    v_max = np.max(roi_hsv[:,2])
    v_min = np.min(roi_hsv[:,2])
    lower = np.array([h_min,s_min,v_min])
    uper = np.array([h_max,s_max,v_max])
    face_mask = cv2.inRange(img_hsv,lower,uper)

    # cv2.namedWindow("smallImg",2)
    # cv2.imshow("smallImg",face_mask)
    # cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    face_mask = cv2.morphologyEx(face_mask,cv2.MORPH_CLOSE,kernel)
    contours, hierarchy = cv2.findContours(face_mask, mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)

    # _,contours, hierarchy = cv2.findContours(face_mask, mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
    if len(contours)!=0:
        c = contours[0]
        c_temp=np.array(c).reshape(-1,2)
        Y_max = (LeftEyeBrow[6]['Y']+RightEyeBrow[6]['Y'])/2
        # Y_max = LeftEyeBrow[6]['Y']
        forehandPoints = c_temp[np.where(c_temp[:,1]<Y_max)]

        if np.shape(forehandPoints)[0] < 10:
            return False,[]

        forehandPoints = forehandPoints[np.argsort(forehandPoints[:,0])]

        # forehandPoints_draw = forehandPoints.reshape(-1,1,2)
        x = forehandPoints[:,0]
        y = forehandPoints[:,1]
        reg=np.polyfit(x,y,5)
        ry=np.polyval(reg,x)
        Points = np.hstack((x.reshape(-1,1),ry.reshape(-1,1))).astype(np.int)

        # img_temp=img.copy()
        # img_temp = cv2.polylines(img_temp, [np.array(Points)],True,(0,0,255),3,lineType = cv2.LINE_8)
        # cv2.namedWindow("smallImg3",2)
        # cv2.imshow("smallImg3",img_temp)
        # cv2.waitKey(0)

        # points_num,_=np.shape(forehandPoints)
        # img_show = img.copy()
        # cv2.drawContours(img_show, forehandPoints_draw, -1, (0, 0, 255), 10)
        # for i in range(points_num):
        #     img=cv2.circle(img_show,(forehandPoints[i,0],forehandPoints[i,1]),1,(0,0,255),1)
        #     cv2.namedWindow("smallImg3",2)
        #     cv2.imshow("smallImg3",img_show)
        #     cv2.waitKey(0)
        # img2=cv2.circle(img_show,(LeftEyeBrow[6]['X'],LeftEyeBrow[6]['Y']),5,(0,0,255),2)
        # cv2.namedWindow("smallImg3",2)
        # cv2.imshow("smallImg3",img2)
        # cv2.waitKey(0)
        # cv2.namedWindow("smallImg3",2)
        # cv2.imshow("smallImg3",img_show)
        # cv2.waitKey(0)
        return True,Points
    else :
        return False,[]

def formatChangeSeeta(pointsL,img):
    '''
    格式更改
    [[x,y],[x,y]]->FaceProfile,LeftEye...
    '''
    FaceProfile=[]
    LeftEye=[]
    RightEye=[]
    LeftEyeBrow=[]
    RightEyeBrow=[]
    Mouth=[]
    Nose=[]
    LeftPupil=[]
    RightPupil=[]
    tempPoint={}
    imgd=img.copy()
    for i,points in enumerate(pointsL):
        tempPoint['X']=int(points[0])
        tempPoint['Y']=int(points[1])
        if i==0:#LeftPupil 1
            LeftPupil.append(tempPoint)
        elif i<=8:#LeftEye 8
            LeftEye.append(tempPoint)
        elif i<=9:#RightPupil 1
            RightPupil.append(tempPoint)
        elif i<=17:#RightEye 8
            RightEye.append(tempPoint)
        elif i<=25:#LeftEyeBrow 8
            LeftEyeBrow.append(tempPoint)
        elif i<=33:#RightEyeBrow 8
            RightEyeBrow.append(tempPoint)
        elif i<=45:#Nose 12
            Nose.append(tempPoint)
        elif i<=59:#Mouse 14
            Mouth.append(tempPoint)
        else:
            FaceProfile.append(tempPoint)
        tempPoint={}
    result={}
    FaceShapeSet=[]
    tempDict={}
    tempDict['FaceProfile']=FaceProfile
    tempDict['LeftEye']=LeftEye
    tempDict['RightEye']=RightEye
    tempDict['LeftEyeBrow']=LeftEyeBrow
    tempDict['RightEyeBrow']=RightEyeBrow
    tempDict['Mouth']=Mouth
    tempDict['Nose']=Nose
    tempDict['LeftPupil']=LeftPupil
    tempDict['RightPupil']=RightPupil
    FaceShapeSet.append(tempDict)
    result['FaceShapeSet']=FaceShapeSet
    return result


def acneSample(img,result,draw=False):
    '''
    img: color
    FaceProfile 所有点
    Mouth 外面一圈
    '''
    FaceProfile=result['FaceShapeSet'][0]['FaceProfile']
    Mouth=result['FaceShapeSet'][0]['Mouth']
    Nose=result['FaceShapeSet'][0]['Nose']
    try:
        flag,forehandPoints=findForeHand_skin(img,result)

        if flag:
            s=50
            x2r=80
            x2l=120
            y2u=50
            y2d=120
            bottomY=FaceProfile[11]['Y']
            if FaceProfile[11]['Y']<FaceProfile[19]['Y']:
                bottomY=FaceProfile[19]['Y']
            FacePoints=np.array(
            [
            [FaceProfile[0]['X']+x2r,FaceProfile[0]['Y']],
            [FaceProfile[2]['X']+x2r,FaceProfile[2]['Y']],
            [FaceProfile[5]['X']+x2r,FaceProfile[5]['Y']],
            [FaceProfile[6]['X']+x2r,FaceProfile[6]['Y']],
            [FaceProfile[7]['X']+x2r,FaceProfile[7]['Y']],
            [FaceProfile[8]['X']+x2r,FaceProfile[8]['Y']],
            [FaceProfile[9]['X']+x2r,FaceProfile[9]['Y']],
            [FaceProfile[10]['X']+x2r,FaceProfile[10]['Y']],
            [FaceProfile[11]['X']+x2r,bottomY-y2u],
            [FaceProfile[12]['X'],bottomY-y2u],
            [FaceProfile[4]['X'],bottomY-y2u],
            [FaceProfile[20]['X'],bottomY-y2u],
            [FaceProfile[19]['X']-x2l,bottomY-y2u],
            [FaceProfile[18]['X']-x2l,FaceProfile[18]['Y']],
            [FaceProfile[17]['X']-x2l,FaceProfile[17]['Y']],
            [FaceProfile[16]['X']-x2l,FaceProfile[16]['Y']],
            [FaceProfile[15]['X']-x2l,FaceProfile[15]['Y']],
            [FaceProfile[14]['X']-x2l,FaceProfile[14]['Y']],
            [FaceProfile[13]['X']-x2l,FaceProfile[13]['Y']],
            [FaceProfile[3]['X']-x2l,FaceProfile[3]['Y']],
            [FaceProfile[1]['X']-x2l,FaceProfile[1]['Y']],
            ]
            )
            FacePoints = FacePoints[::-1]
            ydis=100
            forehandPoints = forehandPoints +[0,ydis]
            Points = np.vstack((forehandPoints,FacePoints))

            MouthPoints=np.array([
            [Mouth[0]['X']-s,Mouth[0]['Y']],
            [Mouth[12]['X'],Mouth[12]['Y']],
            [Mouth[9]['X'],Mouth[9]['Y']],
            [Mouth[13]['X'],Mouth[13]['Y']],
            [Mouth[1]['X'],Mouth[1]['Y']],
            [Mouth[5]['X'],Mouth[5]['Y']],
            [Mouth[2]['X']+s,Mouth[2]['Y']],
            [Mouth[4]['X'],Mouth[4]['Y']]
            ])

            PointsL=findnaris(img,result)
            h,w,_=img.shape
            mask = np.zeros((h,w), dtype='uint8')
            mask = cv2.fillPoly(mask, np.int32([Points]), (255, 255, 255))
            mask = cv2.fillPoly(mask, np.int32([MouthPoints]), (0, 0, 0))
            PointsLR=[Points,MouthPoints]
            for point_t in PointsL:
                # print(np.shape(point_t))
                mask = cv2.fillPoly(mask, np.int32([point_t]), (0, 0, 0))
                PointsLR.append(point_t)
            Indices=np.nonzero(mask)

            if draw:
                img_temp=img.copy()
                img_temp = cv2.polylines(img_temp, [np.array(Points)],True,(0,0,255),3,lineType = cv2.LINE_8)
                img_temp = cv2.polylines(img_temp, [np.array(MouthPoints)],True,(0,0,255),3,lineType = cv2.LINE_8)
                for point_t in PointsL:
                    # print(np.shape(point_t))
                    img_temp = cv2.polylines(img_temp, [np.array(point_t)],True,(0,0,255),3,lineType = cv2.LINE_8)

                cv2.namedWindow("smallImg",2)
                cv2.imshow("smallImg",img_temp)
                cv2.waitKey(0)
                text="demo"
                cv2.putText(img_temp, text, (200,3000), cv2.FONT_HERSHEY_COMPLEX,10, (100, 200, 200), 5)
                cv2.imwrite("D:\\DeskTop\\azure\\new_pic\\2\\acneSample.jpg",img_temp)

                img_temp2=img.copy()
                img_temp2=drawPointsArray(img_temp2,Points)
                cv2.namedWindow("smallImg",2)
                cv2.imshow("smallImg",img_temp2)
                cv2.waitKey(0)

            # cv2.imwrite("acne.jpg",img_temp)

            return mask,PointsLR,len(Indices[0])
        else :
            raise Exception
    except Exception as e :
        print(traceback.format_exc())
        s=50
        x2r=50
        x2l=50
        y2u=130
        y2d=120
        forehand=findForeHand(img)
        FacePoints=np.array(
        [[forehand[0][0],forehand[0][1]+y2d],
        [FaceProfile[0]['X']+x2r,FaceProfile[0]['Y']],
        [FaceProfile[1]['X']+x2r,FaceProfile[1]['Y']],
        [FaceProfile[2]['X']+x2r,FaceProfile[2]['Y']],
        [FaceProfile[3]['X']+x2r,FaceProfile[3]['Y']],
        [FaceProfile[4]['X']+x2r,FaceProfile[4]['Y']],
        [FaceProfile[5]['X']+x2r,FaceProfile[5]['Y']],
        [FaceProfile[6]['X']+x2r,FaceProfile[6]['Y']],
        [FaceProfile[7]['X']+x2r,FaceProfile[7]['Y']],
        [FaceProfile[8]['X'],FaceProfile[8]['Y']],
        [FaceProfile[9]['X'],FaceProfile[9]['Y']-int(0.5*y2u)],
        [FaceProfile[10]['X'],FaceProfile[10]['Y']-y2u],
        [FaceProfile[11]['X'],FaceProfile[11]['Y']-int(0.5*y2u)],
        [FaceProfile[12]['X'],FaceProfile[12]['Y']],
        [FaceProfile[13]['X']-x2l,FaceProfile[13]['Y']],
        [FaceProfile[14]['X']-x2l,FaceProfile[14]['Y']],
        [FaceProfile[15]['X']-x2l,FaceProfile[15]['Y']],
        [FaceProfile[16]['X']-x2l,FaceProfile[16]['Y']],
        [FaceProfile[17]['X']-x2l,FaceProfile[17]['Y']],
        [FaceProfile[18]['X']-x2l,FaceProfile[18]['Y']],
        [FaceProfile[19]['X']-x2l,FaceProfile[19]['Y']],
        [FaceProfile[20]['X']-x2l,FaceProfile[20]['Y']],
        [forehand[1][0],forehand[1][1]+y2d]]
        )
        MouthPoints=np.array([
        [Mouth[0]['X']-s,Mouth[0]['Y']],
        [Mouth[1]['X'],Mouth[1]['Y']+s],
        [Mouth[2]['X'],Mouth[2]['Y']+s],
        [Mouth[3]['X'],Mouth[3]['Y']+s],
        [Mouth[4]['X'],Mouth[4]['Y']+s],
        [Mouth[5]['X'],Mouth[5]['Y']+s],
        [Mouth[6]['X']+s,Mouth[6]['Y']],
        [Mouth[7]['X'],Mouth[7]['Y']-s],
        [Mouth[8]['X'],Mouth[8]['Y']-s],
        [Mouth[9]['X'],Mouth[9]['Y']-s],
        [Mouth[10]['X'],Mouth[10]['Y']-s],
        [Mouth[11]['X'],Mouth[11]['Y']-s]
        ])

        PointsL=findnaris(img,result)
        h,w,_=img.shape
        mask = np.zeros((h,w), dtype='uint8')
        mask = cv2.fillPoly(mask, np.int32([FacePoints]), (255, 255, 255))
        mask = cv2.fillPoly(mask, np.int32([MouthPoints]), (0, 0, 0))
        PointsLR = [FacePoints,MouthPoints]
        for point_t in PointsL:
            # print(np.shape(point_t))
            mask = cv2.fillPoly(mask, np.int32([point_t]), (0, 0, 0))
            PointsLR.append(point_t)
        Indices=np.nonzero(mask)
        return mask,PointsLR,len(Indices[0])

def porphyinSample(img,result,draw=False):
    '''
    鼻头区域
    '''
    FaceProfile=result['FaceShapeSet'][0]['FaceProfile']
    Mouth=result['FaceShapeSet'][0]['Mouth']
    Nose=result['FaceShapeSet'][0]['Nose']
    LeftEye=result['FaceShapeSet'][0]['LeftEye']
    RightEye=result['FaceShapeSet'][0]['RightEye']
    LeftEyeBrow=result['FaceShapeSet'][0]['LeftEyeBrow']
    RightEyeBrow=result['FaceShapeSet'][0]['RightEyeBrow']
    try:
        flag,forehandPoints=findForeHand_skin(img,result)
        if flag:
            s=50
            x2r=50
            x2l=50
            y2u_eyebrow=130
            y2d_eye=120
            x2l_face=140
            x2r_face=130
            x2l_mouth=200
            x2r_mouth=200
            x2l_brow = 80
            x2r_brow = 80
            x2l_nose = 80
            x2r_nose = 80
            y2u_nose = 80


            #左眉毛
            x1=np.array([LeftEyeBrow[0]['X'],LeftEyeBrow[4]['X'],
            LeftEyeBrow[2]['X'],LeftEyeBrow[6]['X'],
            LeftEyeBrow[1]['X']+x2r_brow*2]
            )

            y1=np.array([LeftEyeBrow[0]['Y']-y2u_eyebrow,LeftEyeBrow[4]['Y']-y2u_eyebrow,
            LeftEyeBrow[2]['Y']-y2u_eyebrow,LeftEyeBrow[6]['Y']-y2u_eyebrow,
            LeftEyeBrow[1]['Y']-y2u_eyebrow])

            #左鼻梁
            x2 = np.array([LeftEyeBrow[1]['X']+x2r_brow*2,Nose[2]['X']-x2l_nose,
            Nose[4]['X']-x2l_nose])
            y2 = np.array([LeftEyeBrow[1]['Y']-y2u_eyebrow,Nose[2]['Y'],
            Nose[4]['Y']-y2u_nose])

            #鼻头
            x3 = np.array([Nose[4]['X']-x2l_nose,Nose[0]['X'],Nose[5]['X']+x2r_nose])
            y3 = np.array([Nose[4]['Y']-y2u_nose,Nose[0]['Y'],Nose[5]['Y']-y2u_nose])

            #右鼻梁
            x4 = np.array([RightEyeBrow[0]['X']-x2l_brow*2,Nose[3]['X']+x2r_nose,
            Nose[5]['X']+x2r_nose])
            y4 = np.array([RightEyeBrow[0]['Y']-y2u_eyebrow,Nose[3]['Y'],Nose[5]['Y']-y2u_nose])

            #右眉毛
            x5=np.array([RightEyeBrow[0]['X']-x2l_brow*2,RightEyeBrow[4]['X'],
            RightEyeBrow[2]['X'],RightEyeBrow[6]['X'],
            RightEyeBrow[1]['X']]
            )

            y5=np.array([RightEyeBrow[0]['Y']-y2u_eyebrow,RightEyeBrow[4]['Y']-y2u_eyebrow,
            RightEyeBrow[2]['Y']-y2u_eyebrow,RightEyeBrow[6]['Y']-y2u_eyebrow,
            RightEyeBrow[1]['Y']-y2u_eyebrow])

            x1new,y1new=makeCure(x1,y1,pointsNum=30)
            x2new,y2new=makeCure(x2,y2,pointsNum=30)
            x3new,y3new=makeCure(x3,y3,pointsNum=50)
            x4new,y4new=makeCure(x4,y4,pointsNum=30)
            x5new,y5new=makeCure(x5,y5,pointsNum=30)

            points1t=np.vstack((x1,y1)).T
            points1t=points1t.astype(np.int)
            points1=np.vstack((x1new,y1new)).T
            points1=points1.astype(np.int)


            points2t=np.vstack((x2,y2)).T
            points2t=points2t.astype(np.int)
            points2=np.vstack((x2new,y2new)).T
            points2=points2.astype(np.int)

            points3t=np.vstack((x3,y3)).T
            points3t=points3t.astype(np.int)
            points3=np.vstack((x3new,y3new)).T
            points3=points3.astype(np.int)

            points4t=np.vstack((x4,y4)).T
            points4t=points4t.astype(np.int)
            points4=np.vstack((x4new,y4new)).T
            points4=points4.astype(np.int)

            points5t=np.vstack((x5,y5)).T
            points5t=points5t.astype(np.int)
            points5=np.vstack((x5new,y5new)).T
            points5=points5.astype(np.int)


            #额头部分
            ydis = 60
            forehandPoints = forehandPoints + [0,ydis]
            forehandPoints = forehandPoints[::-1]

            forehandPoints = forehandPoints[np.where(forehandPoints[:,0]>LeftEyeBrow[0]['X'])]
            forehandPoints = forehandPoints[np.where(forehandPoints[:,0]<RightEyeBrow[1]['X'])]

            points2 = points2[::-1]
            pointsMiddle = np.vstack((points1,points2))
            pointsMiddle = np.vstack((pointsMiddle,points3))
            points4 = points4[::-1]
            pointsMiddle = np.vstack((pointsMiddle,points4))
            pointsMiddle = np.vstack((pointsMiddle,points5))
            pointsMiddle = np.vstack((pointsMiddle,forehandPoints))
            if draw:
                img_temp=img.copy()
                img_temp = cv2.polylines(img_temp, [np.array(pointsMiddle)],True,(0,0,255),3,lineType = cv2.LINE_8)
                cv2.namedWindow("smallImg",2)
                cv2.imshow("smallImg",img_temp)
                cv2.waitKey(0)

                img_temp2=img.copy()
                img_temp2=drawPointsArray(img_temp2,pointsMiddle)
                cv2.namedWindow("smallImg",2)
                cv2.imshow("smallImg",img_temp2)
                cv2.waitKey(0)

                text="demo"
                cv2.putText(img_temp, text, (200,3000), cv2.FONT_HERSHEY_COMPLEX,10, (100, 200, 200), 5)
                cv2.imwrite("D:\\DeskTop\\azure\\pics\\6\\leftSample.jpg",img_temp)

            h=img.shape[0]
            w=img.shape[1]
            mask = np.zeros((h,w), dtype='uint8')
            mask = cv2.fillPoly(mask, np.int32([pointsMiddle]), (255, 255, 255))
            Indices=np.nonzero(mask)

            return mask,[pointsMiddle],len(Indices[0])
        else :
            raise Exception
    except Exception as e :
        print(traceback.format_exc())
        foreHand=findForeHand(img)
        x2r=70
        x2l=70
        y2u=50
        y2d=80
        ydis=LeftEyeBrow[6]['Y']-foreHand[0][1]-30
        Points=np.array([
        [LeftEyeBrow[7]['X'],LeftEyeBrow[7]['Y']-y2u],
        [LeftEyeBrow[6]['X'], LeftEyeBrow[6]['Y']-y2u],
        [LeftEyeBrow[5]['X'], LeftEyeBrow[5]['Y']-y2u],
        [LeftEyeBrow[4]['X'], LeftEyeBrow[4]['Y']-y2u],
        [Nose[1]['X']-x2l, Nose[1]['Y']],
        [Nose[2]['X']-x2l, Nose[2]['Y']],
        [Nose[3]['X']-x2l, Nose[3]['Y']],
        [Nose[4]['X']-x2l, Nose[4]['Y']],
        [Nose[5]['X'], Nose[5]['Y']],
        [Nose[6]['X'], Nose[6]['Y']],
        [Nose[7]['X'], Nose[7]['Y']],
        [Nose[8]['X'], Nose[8]['Y']],
        [Nose[9]['X'], Nose[9]['Y']],
        [Nose[10]['X']+x2r, Nose[10]['Y']],
        [Nose[11]['X']+x2r, Nose[11]['Y']],
        [Nose[12]['X']+x2r, Nose[12]['Y']],
        [Nose[1]['X']+x2r, Nose[1]['Y']],
        [RightEyeBrow[4]['X'], RightEyeBrow[4]['Y']-y2u],
        [RightEyeBrow[5]['X'],RightEyeBrow[5]['Y']-y2u],
        [RightEyeBrow[6]['X'], RightEyeBrow[6]['Y']-y2u],
        [RightEyeBrow[7]['X'], RightEyeBrow[7]['Y']-y2u],
        [RightEyeBrow[7]['X'], RightEyeBrow[7]['Y']-int(0.5*ydis)],
        [RightEyeBrow[6]['X'], foreHand[0][1]+y2d],
        [RightEyeBrow[5]['X'], foreHand[0][1]+y2d],
        [RightEyeBrow[4]['X'], foreHand[0][1]+y2d],
        [Nose[1]['X'], foreHand[0][1]+y2d],
        [LeftEyeBrow[4]['X'],foreHand[0][1]+y2d],
        [LeftEyeBrow[5]['X'],foreHand[0][1]+y2d],
        [LeftEyeBrow[6]['X'], foreHand[0][1]+y2d],
        [LeftEyeBrow[7]['X'], LeftEyeBrow[7]['Y']-int(0.5*ydis)]])



        PointsL = findnaris(img,result)
        h,w,_=img.shape
        mask = np.zeros((h,w), dtype='uint8')
        mask = cv2.fillPoly(mask, np.int32([Points]), (255, 255, 255))
        PointsLR=[Points]
        for point_t in PointsL:
            mask = cv2.fillPoly(mask, np.int32([point_t]), (0, 0, 0))
            PointsLR.append(point_t)
        Indices=np.nonzero(mask)

        return mask,PointsLR,len(Indices[0])

def leftSample(img,result,draw=False):

    FaceProfile=result['FaceShapeSet'][0]['FaceProfile']
    Mouth=result['FaceShapeSet'][0]['Mouth']
    Nose=result['FaceShapeSet'][0]['Nose']
    LeftEye=result['FaceShapeSet'][0]['LeftEye']
    RightEye=result['FaceShapeSet'][0]['RightEye']
    RightEyeBrow=result['FaceShapeSet'][0]['RightEyeBrow']

    xdis=30
    x2l_face=265
    x2l_nose=30
    x2r_nose=30
    x2r_nose2=40
    x2d_nose=30
    x2r_brow=300
    y2d_eye=200
    y2u_nose=60
    y2u_face=80
    nose_y=int(0.5*(Nose[11]['Y']+Nose[0]['Y']))
    ydis=15
    if nose_y-ydis<Nose[0]['Y']:
        nose_y=Nose[0]['Y']
    else:
        nose_y=nose_y-ydis

    x2l_faceS=int(0.3*(FaceProfile[3]['X']-RightEyeBrow[1]['X']))
#       1
    # x1=np.array([Nose[4]['X'],
    # Nose[2]['X']+x2r_nose2,Nose[3]['X'],
    # RightEye[0]['X'],RightEye[5]['X'],
    # RightEye[3]['X'],RightEye[7]['X'],
    # RightEye[1]['X']+x2r_brow,RightEyeBrow[1]['X']+x2r_brow,
    # FaceProfile[3]['X']-int(2.3*x2l_face),FaceProfile[3]['X']-x2l_face])

    x1=np.array([Nose[4]['X'],
    Nose[2]['X']+x2r_nose2,Nose[3]['X'],
    RightEye[0]['X'],RightEye[5]['X'],
    RightEye[3]['X'],RightEye[7]['X'],
    RightEye[1]['X']+int(0.6*x2l_faceS),RightEyeBrow[1]['X']+int(0.6*x2l_faceS),
    FaceProfile[3]['X']-int(1.7*x2l_faceS),FaceProfile[3]['X']-int(0.7*x2l_faceS)])

    y1=np.array([Nose[4]['Y'],
    Nose[2]['Y'],Nose[3]['Y'],
    RightEye[0]['Y']+y2d_eye,RightEye[5]['Y']+y2d_eye,
    RightEye[3]['Y']+y2d_eye,RightEye[7]['Y']+y2d_eye,
    RightEye[1]['Y'],RightEyeBrow[1]['Y'],
    FaceProfile[1]['Y'],FaceProfile[3]['Y']])


    #嘴巴
    # x2=np.array([Nose[4]['X'],Nose[0]['X']+x2r_nose,
    # Nose[11]['X'],Nose[5]['X'],Nose[7]['X']+x2r_nose*5,FaceProfile[17]['X']-x2l_face,
    # FaceProfile[16]['X']-x2l_face])

    x2=np.array([Nose[4]['X'],Nose[0]['X']+x2r_nose,
    Nose[11]['X'],Nose[5]['X'],Nose[7]['X']+x2r_nose*5,FaceProfile[17]['X']-x2l_face,
    FaceProfile[16]['X']-int(0.7*x2l_faceS)])

    # noseYdpress=int(0.5*(Nose[5]['Y']+Nose[3]['Y']))
    y2=np.array([Nose[4]['Y'],Nose[0]['Y']-y2u_nose,
    nose_y-y2u_nose,Nose[5]['Y']-y2u_nose*2,Nose[7]['Y']-y2u_nose,FaceProfile[17]['Y']-y2u_face,
    FaceProfile[16]['Y']]
    )


    x1new,y1new=makeCure(x1,y1,pointsNum=1000)
    x2new,y2new=makeCure(x2,y2,pointsNum=1000)

    points1t=np.vstack((x1,y1)).T
    points1t=points1t.astype(np.int)
    points1=np.vstack((x1new,y1new)).T
    points1=points1.astype(np.int)


    points2t=np.vstack((x2,y2)).T
    points2t=points2t.astype(np.int)
    points2=np.vstack((x2new,y2new)).T
    points2=points2.astype(np.int)


    # facePoints=np.array([[FaceProfile[13]['X']-x2l_face,FaceProfile[13]['Y']],
    # [FaceProfile[14]['X']-x2l_face,FaceProfile[14]['Y']],
    # [FaceProfile[15]['X']-x2l_face,FaceProfile[15]['Y']],
    # [FaceProfile[16]['X']-x2l_face,FaceProfile[16]['Y']]])

    facePoints=np.array([[FaceProfile[13]['X']-int(0.7*x2l_faceS),FaceProfile[13]['Y']],
    [FaceProfile[14]['X']-int(0.7*x2l_faceS),FaceProfile[14]['Y']],
    [FaceProfile[15]['X']-int(0.7*x2l_faceS),FaceProfile[15]['Y']],
    [FaceProfile[16]['X']-int(0.7*x2l_faceS),FaceProfile[16]['Y']]])

    #鼻子
    pointsNose1=np.array([
    [Nose[4]['X'],Nose[4]['Y']],
    [Nose[2]['X']+x2r_nose2,Nose[2]['Y']]])
    #脸部曲线截取
    leftPoints=points1[np.where(points1[:,0]>Nose[2]['X']+x2r_nose2)]
    #嘴巴 鼻子部分曲线截取
    leftPointsTemp = points2[np.where(points2[:,0]<FaceProfile[16]['X']-int(0.7*x2l_faceS))]
    #拼接
    leftPoints = np.vstack((pointsNose1,leftPoints))
    leftPoints = np.vstack((leftPoints,facePoints))
    leftPointsTemp = leftPointsTemp[::-1]
    leftPoints = np.vstack((leftPoints,leftPointsTemp))

    if draw:
        img_temp=img.copy()
        img_temp = cv2.polylines(img_temp, [np.array(leftPoints)],True,(0,0,255),3,lineType = cv2.LINE_8)
        cv2.namedWindow("smallImg",2)
        cv2.imshow("smallImg",img_temp)
        cv2.waitKey(0)

        img_temp2=img.copy()
        img_temp2=drawPointsArray(img_temp2,leftPoints)
        cv2.namedWindow("smallImg",2)
        cv2.imshow("smallImg",img_temp2)
        cv2.waitKey(0)

        # text="demo"
        # cv2.putText(img_temp, text, (200,3000), cv2.FONT_HERSHEY_COMPLEX,10, (100, 200, 200), 5)
        cv2.imwrite("D:\\DeskTop\\azure\\pics\\6\\leftSample.jpg",img_temp)

    h=img.shape[0]
    w=img.shape[1]
    mask = np.zeros((h,w), dtype='uint8')
    mask = cv2.fillPoly(mask, np.int32([leftPoints]), (255, 255, 255))
    Indices=np.nonzero(mask)

    return mask,[leftPoints],len(Indices[0])

def rightSample(img,result,draw=False):
    h=img.shape[0]
    w=img.shape[1]

    FaceProfile=result['FaceShapeSet'][0]['FaceProfile']
    Mouth=result['FaceShapeSet'][0]['Mouth']
    Nose=result['FaceShapeSet'][0]['Nose']
    LeftEye=result['FaceShapeSet'][0]['LeftEye']
    LeftEyeBrow=result['FaceShapeSet'][0]['LeftEyeBrow']

    x2l_face=230
    x2r_face=250
    x2d_face=50
    x2l_nose=50
    x2l_nose2=40
    x2r_nose=40
    x2d_nose=30
    x2l_brow=300
    y2d_eye=200
    y2u_nose=60
    y2u_face=50



    nose_y=int(0.5*(Nose[10]['Y']+Nose[0]['Y']))
    ydis=15
    if nose_y-ydis<Nose[0]['Y']:
        nose_y=Nose[0]['Y']
    else:
        nose_y=nose_y-ydis


    x1=np.array([Nose[5]['X']+x2r_nose,
    Nose[3]['X'],Nose[2]['X'],
    LeftEye[1]['X'],LeftEye[7]['X'],
    LeftEye[3]['X'],LeftEye[5]['X'],
    LeftEye[0]['X']-x2l_brow,LeftEyeBrow[0]['X']-x2l_brow,
    FaceProfile[2]['X']+2*x2r_face,FaceProfile[2]['X']+x2r_face])

    y1=np.array([Nose[5]['Y'],
    Nose[3]['Y'],Nose[2]['Y'],
    LeftEye[1]['Y']+y2d_eye,LeftEye[7]['Y']+y2d_eye,
    LeftEye[3]['Y']+y2d_eye,LeftEye[5]['Y']+y2d_eye,
    LeftEye[0]['Y'],LeftEyeBrow[0]['Y'],
    FaceProfile[0]['Y'],FaceProfile[2]['Y']])

    #嘴巴


    x2=np.array([Nose[5]['X']+x2r_nose,Nose[0]['X'],
    Nose[10]['X'],Nose[4]['X'],Nose[6]['X']-x2l_nose*5,FaceProfile[9]['X']+x2r_face,
    FaceProfile[8]['X']+x2r_face])

    y2=np.array([Nose[5]['Y'],Nose[0]['Y']-y2u_nose,
    nose_y-y2u_nose,Nose[4]['Y']-y2u_nose*2,Nose[6]['Y']-y2u_nose,FaceProfile[9]['Y']-y2u_face,
    FaceProfile[8]['Y']]
    )


    x1new,y1new=makeCure(x1,y1,pointsNum=100)
    x2new,y2new=makeCure(x2,y2,pointsNum=100)

    points1t=np.vstack((x1,y1)).T
    points1t=points1t.astype(np.int)
    points1=np.vstack((x1new,y1new)).T
    points1=points1.astype(np.int)


    points2t=np.vstack((x2,y2)).T
    points2t=points2t.astype(np.int)
    points2=np.vstack((x2new,y2new)).T
    points2=points2.astype(np.int)



    facePoints=np.array([[FaceProfile[2]['X']+x2r_face,FaceProfile[2]['Y']],
    [FaceProfile[5]['X']+x2r_face,FaceProfile[5]['Y']],
    [FaceProfile[6]['X']+x2r_face,FaceProfile[6]['Y']],
    [FaceProfile[7]['X']+x2r_face,FaceProfile[7]['Y']],
    [FaceProfile[8]['X']+x2r_face,FaceProfile[8]['Y']]])

    #鼻子
    pointsNose1=np.array([
    [Nose[3]['X'],Nose[3]['Y']],
    [Nose[5]['X']+x2r_nose,Nose[5]['Y']]])
    #脸部曲线截取
    leftPoints=points1[np.where(points1[:,0]<Nose[3]['X'])]
    #嘴巴 鼻子部分曲线截取
    leftPointsTemp = points2[np.where(points2[:,0]>FaceProfile[8]['X']+x2r_face)]
    #拼接
    leftPoints = np.vstack((leftPoints,pointsNose1))
    leftPointsTemp = leftPointsTemp[::-1]
    leftPoints = np.vstack((leftPoints,leftPointsTemp))
    facePoints = facePoints[::-1]
    leftPoints = np.vstack((leftPoints,facePoints))

    mask = np.zeros((h,w), dtype='uint8')
    mask = cv2.fillPoly(mask, np.int32([leftPoints]), (255, 255, 255))
    Indices=np.nonzero(mask)



    if draw:
        img_temp=img.copy()
        img_temp = cv2.polylines(img_temp, [np.array(leftPoints)],True,(0,0,255),3,lineType = cv2.LINE_8)
        cv2.namedWindow("smallImg",2)
        cv2.imshow("smallImg",img_temp)
        cv2.waitKey(0)

        img_temp2=img.copy()
        img_temp2=drawPointsArray(img_temp2,points2t)
        cv2.namedWindow("smallImg",2)
        cv2.imshow("smallImg",img_temp2)
        cv2.waitKey(0)

        text="demo"
        cv2.putText(img_temp, text, (200,3000), cv2.FONT_HERSHEY_COMPLEX,10, (100, 200, 200), 5)
        cv2.imwrite("D:\\DeskTop\\azure\\pics\\6\\rightSample.jpg",img_temp)

    return mask,[leftPoints],len(Indices[0])

def middleSample(img,result,draw=False):
    FaceProfile=result['FaceShapeSet'][0]['FaceProfile']
    Mouth=result['FaceShapeSet'][0]['Mouth']
    Nose=result['FaceShapeSet'][0]['Nose']
    LeftEye=result['FaceShapeSet'][0]['LeftEye']
    RightEye=result['FaceShapeSet'][0]['RightEye']
    LeftEyeBrow=result['FaceShapeSet'][0]['LeftEyeBrow']
    RightEyeBrow=result['FaceShapeSet'][0]['RightEyeBrow']
    try:
        flag,forehandPoints=findForeHand_skin(img,result)
        if flag:
            s=50
            x2r=50
            x2l=50
            y2u_eyebrow=130
            y2d_eye=120
            x2l_face=140
            x2r_face=130
            x2l_mouth=200
            x2r_mouth=200
            x2l_brow = 70
            x2r_brow = 70
            x2l_nose = 80
            x2r_nose = 80
            y2u_nose = 80


            #左眉毛
            x1=np.array([LeftEyeBrow[0]['X'],LeftEyeBrow[4]['X'],
            LeftEyeBrow[2]['X'],LeftEyeBrow[6]['X'],
            LeftEyeBrow[1]['X']+x2r_brow*2]
            )

            y1=np.array([LeftEyeBrow[0]['Y']-y2u_eyebrow,LeftEyeBrow[4]['Y']-y2u_eyebrow,
            LeftEyeBrow[2]['Y']-y2u_eyebrow,LeftEyeBrow[6]['Y']-y2u_eyebrow,
            LeftEyeBrow[1]['Y']-y2u_eyebrow])

            #左鼻梁
            x2 =np.array([LeftEyeBrow[2]['X'],LeftEyeBrow[6]['X'],
            LeftEyeBrow[1]['X']+x2r_brow*2,Nose[2]['X']-x2l_nose+x2r_nose*2])

            y2 = np.array([LeftEyeBrow[2]['Y']-y2u_eyebrow,LeftEyeBrow[6]['Y']-y2u_eyebrow,
            LeftEyeBrow[1]['Y']-y2u_eyebrow,Nose[2]['Y']])


            #眼下沿
            x3 = np.array([LeftEye[1]['X'],LeftEye[7]['X'],
            LeftEye[3]['X'],LeftEye[5]['X'],
            LeftEye[0]['X'],FaceProfile[5]['X']+x2r_face])

            y3 = np.array([LeftEye[1]['Y']+y2d_eye,LeftEye[7]['Y']+y2d_eye,
            LeftEye[3]['Y']+y2d_eye,LeftEye[5]['Y']+y2d_eye,
            LeftEye[0]['Y']+y2d_eye,FaceProfile[5]['Y']])

            #鼻子及两侧
            x4 = np.array([FaceProfile[9]['X']+x2r_face,Mouth[0]['X']-x2l_mouth,
            Nose[6]['X']-x2l_nose,Nose[4]['X']-x2l_nose,Nose[0]['X'],Nose[5]['X']+x2r_nose,
            Nose[7]['X']+x2r_nose,Mouth[1]['X']+x2r_mouth,
            int(0.5*(FaceProfile[16]['X']+Mouth[1]['X']))])

            y4 = np.array([Mouth[0]['Y'],Mouth[0]['Y'],
            Nose[6]['Y'],Nose[4]['Y']-y2u_nose,Nose[0]['Y'],Nose[5]['Y']-y2u_nose,
            Nose[7]['Y'],Mouth[1]['Y'],
            Mouth[1]['Y']])

            #右眼下沿
            x5 = np.array([RightEye[1]['X'],RightEye[7]['X'],
            RightEye[3]['X'],RightEye[5]['X'],
            RightEye[0]['X'],FaceProfile[13]['X']-x2l_face])

            y5 = np.array([RightEye[1]['Y']+y2d_eye,RightEye[7]['Y']+y2d_eye,
            RightEye[3]['Y']+y2d_eye,RightEye[5]['Y']+y2d_eye,
            RightEye[0]['Y']+y2d_eye,FaceProfile[13]['Y']])

            #右鼻梁
            x6 = np.array([Nose[3]['X']+x2r_nose-x2l_nose*2,RightEyeBrow[0]['X']-x2l_brow*2,
            RightEyeBrow[4]['X'],RightEyeBrow[2]['X']])

            y6 = np.array([Nose[3]['Y'],RightEyeBrow[0]['Y']-y2u_eyebrow,
            RightEyeBrow[4]['Y']-y2u_eyebrow,RightEyeBrow[2]['Y']-y2u_eyebrow])

            #右眉毛
            x7=np.array([RightEyeBrow[0]['X']-x2l_brow*2,RightEyeBrow[4]['X'],
            RightEyeBrow[2]['X'],RightEyeBrow[6]['X'],
            RightEyeBrow[1]['X']]
            )

            y7=np.array([RightEyeBrow[0]['Y']-y2u_eyebrow,RightEyeBrow[4]['Y']-y2u_eyebrow,
            RightEyeBrow[2]['Y']-y2u_eyebrow,RightEyeBrow[6]['Y']-y2u_eyebrow,
            RightEyeBrow[1]['Y']-y2u_eyebrow])

            x1new,y1new=makeCure(x1,y1,pointsNum=30)
            x2new,y2new=makeCure(x2,y2,pointsNum=30)
            x3new,y3new=makeCure(x3,y3,pointsNum=50)
            x4new,y4new=makeCure(x4,y4,pointsNum=200)
            x5new,y5new=makeCure(x5,y5,pointsNum=50)
            x6new,y6new=makeCure(x6,y6,pointsNum=30)
            x7new,y7new=makeCure(x7,y7,pointsNum=30)

            points1t=np.vstack((x1,y1)).T
            points1t=points1t.astype(np.int)
            points1=np.vstack((x1new,y1new)).T
            points1=points1.astype(np.int)


            points2t=np.vstack((x2,y2)).T
            points2t=points2t.astype(np.int)
            points2=np.vstack((x2new,y2new)).T
            points2=points2.astype(np.int)

            points3t=np.vstack((x3,y3)).T
            points3t=points3t.astype(np.int)
            points3=np.vstack((x3new,y3new)).T
            points3=points3.astype(np.int)


            points4t=np.vstack((x4,y4)).T
            points4t=points4t.astype(np.int)
            points4=np.vstack((x4new,y4new)).T
            points4=points4.astype(np.int)

            points5t=np.vstack((x5,y5)).T
            points5t=points5t.astype(np.int)
            points5=np.vstack((x5new,y5new)).T
            points5=points5.astype(np.int)

            points6t=np.vstack((x6,y6)).T
            points6t=points6t.astype(np.int)
            points6=np.vstack((x6new,y6new)).T
            points6=points6.astype(np.int)

            points7t=np.vstack((x7,y7)).T
            points7t=points7t.astype(np.int)
            points7=np.vstack((x7new,y7new)).T
            points7=points7.astype(np.int)

            #拼接
            #鼻梁部分截取
            pointsLeftNose = points2[np.where(points2[:,0]>LeftEyeBrow[1]['X']+x2r_brow*2)]
            pointsRightNose = points6[np.where(points6[:,0]<RightEyeBrow[0]['X']-x2l_brow*2)]
            #左边脸轮廓
            pointsLeftFace = np.array([[FaceProfile[5]['X']+x2r_face,FaceProfile[5]['Y']],
            [FaceProfile[6]['X']+x2r_face,FaceProfile[6]['Y']],
            [FaceProfile[7]['X']+x2r_face,FaceProfile[7]['Y']],
            [FaceProfile[8]['X']+x2r_face,FaceProfile[8]['Y']],
            [FaceProfile[9]['X']+x2r_face,Mouth[0]['Y']],
            ])
            #右边脸轮廓
            pointsRightFace = np.array([[int(0.5*(FaceProfile[16]['X']+Mouth[1]['X'])),Mouth[1]['Y']],
            [FaceProfile[16]['X']-x2l_face,FaceProfile[16]['Y']],
            [FaceProfile[15]['X']-x2l_face,FaceProfile[15]['Y']],
            [FaceProfile[14]['X']-x2l_face,FaceProfile[14]['Y']],
            [FaceProfile[13]['X']-x2l_face,FaceProfile[13]['Y']],
            ])

            pointsMiddle = np.vstack((points1,pointsLeftNose))
            points3 = points3[::-1]
            pointsMiddle = np.vstack((pointsMiddle,points3))
            pointsMiddle = np.vstack((pointsMiddle,pointsLeftFace))
            pointsMiddle = np.vstack((pointsMiddle,points4))
            pointsMiddle = np.vstack((pointsMiddle,pointsRightFace))
            points5 = points5[::-1]
            pointsMiddle = np.vstack((pointsMiddle,points5))
            pointsMiddle = np.vstack((pointsMiddle,pointsRightNose))
            pointsMiddle = np.vstack((pointsMiddle,points7))

            #额头部分
            ydis = 60
            forehandPoints = forehandPoints + [0,ydis]
            forehandPoints = forehandPoints[::-1]

            forehandPoints = forehandPoints[np.where(forehandPoints[:,0]>LeftEyeBrow[0]['X'])]
            forehandPoints = forehandPoints[np.where(forehandPoints[:,0]<RightEyeBrow[1]['X'])]
            pointsMiddle = np.vstack((pointsMiddle,forehandPoints))


            if draw:
                img_temp=img.copy()
                img_temp = cv2.polylines(img_temp, [np.array(pointsMiddle)],True,(0,0,255),3,lineType = cv2.LINE_8)
                # for point_t in PointsL:
                #     img_temp = cv2.polylines(img_temp, [np.array(point_t)],True,(0,0,255),3,lineType = cv2.LINE_8)
                cv2.namedWindow("smallImg",2)
                cv2.imshow("smallImg",img_temp)
                cv2.waitKey(0)
                # text="demo"
                # cv2.putText(img_temp, text, (200,3000), cv2.FONT_HERSHEY_COMPLEX,10, (100, 200, 200), 5)
                # cv2.imwrite("D:\\DeskTop\\azure\\pics\\5\\middleSample.jpg",img_temp)

                img_temp2=img.copy()
                img_temp2=drawPointsArray(img_temp2,pointsMiddle)
                cv2.namedWindow("smallImg",2)
                cv2.imshow("smallImg",img_temp2)
                cv2.waitKey(0)
            h,w,_=img.shape
            mask = np.zeros((h,w), dtype='uint8')
            mask = cv2.fillPoly(mask, np.int32([pointsMiddle]), (255, 255, 255))
            Indices=np.nonzero(mask)
            return mask,[pointsMiddle],len(Indices[0])
        else :
            # raise Exception
            print("find forehand failed")
    except Exception as e :
        print(traceback.format_exc())

def FaceSample(img,result,draw=False):
    FaceProfile=result['FaceShapeSet'][0]['FaceProfile']
    try:
        flag,forehandPoints=findForeHand_skin(img,result)
        if flag:
            x2l_face=110
            x2r_face=110

            x1=np.array([FaceProfile[5]['X']+x2l_face,FaceProfile[6]['X']+x2l_face,
            FaceProfile[7]['X']+x2l_face,FaceProfile[8]['X']+x2l_face,
            FaceProfile[9]['X']+x2l_face,FaceProfile[10]['X']+x2l_face,
            FaceProfile[11]['X']+x2l_face]
            )

            y1=np.array([FaceProfile[5]['Y'],FaceProfile[6]['Y'],
            FaceProfile[7]['Y'],FaceProfile[8]['Y'],
            FaceProfile[9]['Y'],FaceProfile[10]['Y'],
            FaceProfile[11]['Y']])

            x2=np.array([
            FaceProfile[19]['X']-x2r_face,FaceProfile[18]['X']-x2r_face,
            FaceProfile[17]['X']-x2r_face,FaceProfile[16]['X']-x2r_face,
            FaceProfile[15]['X']-x2r_face,FaceProfile[14]['X']-x2r_face,
            FaceProfile[13]['X']-x2r_face]
            )

            y2=np.array([
            FaceProfile[19]['Y'],FaceProfile[18]['Y'],
            FaceProfile[17]['Y'],FaceProfile[16]['Y'],
            FaceProfile[15]['Y'],FaceProfile[14]['Y'],
            FaceProfile[13]['Y']])

            x1new,y1new=makeCure(x1,y1,pointsNum=30)
            x2new,y2new=makeCure(x2,y2,pointsNum=30)


            points1t=np.vstack((x1,y1)).T
            points1t=points1t.astype(np.int)
            points1=np.vstack((x1new,y1new)).T
            points1=points1.astype(np.int)


            points2t=np.vstack((x2,y2)).T
            points2t=points2t.astype(np.int)
            points2=np.vstack((x2new,y2new)).T
            points2=points2.astype(np.int)

            pointsMiddle = np.vstack((points1,points2))

            ydis = 60
            forehandPoints = forehandPoints + [0,ydis]
            forehandPoints = forehandPoints[::-1]

            forehandPoints = forehandPoints[np.where(forehandPoints[:,0]>FaceProfile[5]['X']+x2l_face)]
            forehandPoints = forehandPoints[np.where(forehandPoints[:,0]<FaceProfile[13]['X']-x2r_face)]
            pointsMiddle = np.vstack((pointsMiddle,forehandPoints))
            if draw:
                img_temp=img.copy()
                img_temp = cv2.polylines(img_temp, [np.array(pointsMiddle)],True,(0,0,255),3,lineType = cv2.LINE_8)
                # for point_t in PointsL:
                #     img_temp = cv2.polylines(img_temp, [np.array(point_t)],True,(0,0,255),3,lineType = cv2.LINE_8)
                cv2.namedWindow("smallImg",2)
                cv2.imshow("smallImg",img_temp)
                cv2.waitKey(0)
                # text="demo"
                # cv2.putText(img_temp, text, (200,3000), cv2.FONT_HERSHEY_COMPLEX,10, (100, 200, 200), 5)
                # cv2.imwrite("D:\\DeskTop\\azure\\pics\\5\\middleSample.jpg",img_temp)

                img_temp2=img.copy()
                img_temp2=drawPointsArray(img_temp2,pointsMiddle)
                cv2.namedWindow("smallImg",2)
                cv2.imshow("smallImg",img_temp2)
                cv2.waitKey(0)
            h,w,_=img.shape
            mask = np.zeros((h,w), dtype='uint8')
            mask = cv2.fillPoly(mask, np.int32([pointsMiddle]), (255, 255, 255))
            Indices=np.nonzero(mask)
            return mask,[pointsMiddle],len(Indices[0])
        else :
            # raise Exception
            print("find forehand failed")
    except Exception as e :
        print(traceback.format_exc())
def SampleWapper(img,result,direction,draw=False):
    if direction =="left":
        return leftSample(img=img,result=result,draw=draw)
    elif direction =="right":
        return rightSample(img,result,draw=draw)
    elif direction == "middle":
        return middleSample(img,result,draw=draw)
    else:
        print("direction error")

def makeCure(x,y,pointsNum=100,kind='quadratic',show=False):
    '''
    x:np.array (m,)
    y:np.array (n,)
    pointsNum:插值点数
    kind: quadratic linear cubic
    '''
    xnew =np.linspace(x.min(),x.max(),pointsNum)

    #实现函数
    func = interpolate.interp1d(x,y,kind=kind,bounds_error=False)

    #利用xnew和func函数生成ynew,xnew数量等于ynew数量
    ynew = func(xnew)
    if show:
        import matplotlib.pyplot as plt
        plt.plot(x, y, "r", linewidth=1)
        plt.plot(xnew,ynew)
        plt.show()
    return xnew,ynew

def heic2jpeg(source,destination):
    onlyfiles =[f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
    if not os.path.exists(destination):
        os.makedirs(destination)
    else:
        print("Output dir already exists")
    for file in onlyfiles:
        (name, ext) = os.path.basename(file).split('.')
        if ext == 'heic' or ext == 'HEIC':
            dest = os.path.join(destination, name) + '.jpg'
            sour= os.path.join(source, file)
            f=open(sour,"rb")
            bytesIo=f.read()
            fmt = whatimage.identify_image(bytesIo)
            if fmt in ['heic', 'avif']:
                try:
                    i = pyheif.read_heif(bytesIo)
                except pyheif.error.HeifError as e:
                    #raise TypeError from e
                    raise TypeError
             # Extract metadata etc
                for metadata in i.metadata or []:
                    if metadata['type']=='Exif':
                         # do whatever
                         print("EXif")
                 # Convert to other file format like jpeg
                s = io.BytesIO()
                pi = Image.frombytes(
                        mode=i.mode, size=i.size, data=i.data)
                # help(pi.save)
                # pi.save(s, format="jpeg")
                pi.save(dest,quality=95)




if __name__ == '__main__':
    from face import FaceDetectorSeeta

    # imgPath = "D:/DeskTop/1.jpg"
    # img = cv2.imread(imgPath)
    # Detector = FaceDetectorSeeta()
    # _,result= Detector.detect(img)
    # faceMask, pointLists, totalArea=SampleWapper(img,result,direction="middle",draw=True)
    # drawPoints(img,result)

    testImgPath="D:\\svnProject\\cskin\\new_cskin_api\\src\\cskin-analysis-api\\src\\features\\assets\\TestImg"
    n=10

    #调参代码
    prefix=0
    Pre="left_white"
    # Pre="right_white"
    # Pre ="middle_white"
    # one_pic=True
    one_pic=False
    pic_num=10
    Detector = FaceDetectorSeeta()
    for i in range(n):
        if one_pic:
            imgName=Pre+str(pic_num)+".jpg"
            imgPath=os.path.join(testImgPath,imgName)
            img=cv2.imread(imgPath)
            # img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            _,result= Detector.detect(img)
            # print(result)
            faceMask, pointLists, totalArea=leftSample(img,result,draw=True)
            # faceMask, pointLists, totalArea=rightSample(img,result,draw=False)
            # faceMask, pointLists, totalArea=middleSample(img,result)
            # porphyinSample(img,result,draw=True)
            # uvChannelSample(img,result)
            imgd=drawPoints(img,result)
            cv2.namedWindow("test1",2)
            cv2.resizeWindow("test1",imgd.shape[0],imgd.shape[1])
            cv2.imshow("test1",imgd)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
        if i==1:
            continue
        if prefix == 1:
            Pre="right_white"
        if prefix ==2:
            Pre = "middle_white"
        imgName=Pre+str(i)+".jpg"
        print("img:",imgName)
        imgPath=os.path.join(testImgPath,imgName)
        img=cv2.imread(imgPath)
        # img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        Detector = FaceDetectorSeeta()
        _,result= Detector.detect(img)
        # print(result)
        faceMask, pointLists, totalArea=leftSample(img,result,draw=False)
        # faceMask, pointLists, totalArea=rightSample(img,result,draw=True)
        # faceMask, pointLists, totalArea=middleSample(img,result,draw=True)
        # uvChannelSample(img,result)
        # faceMask, pointLists, totalArea=porphyinSample(img,result,draw=True)
        # faceMask, pointLists, totalArea=FaceSample(img,result,draw=True)
        # imgd=drawPoints(img,result)
        # cv2.namedWindow("test1",2)
        # cv2.resizeWindow("test1",imgd.shape[0],imgd.shape[1])
        # cv2.imshow("test1",imgd)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    #生成样本代码
    # produceMaskPic(testImgPath,n)


    # poreSample(img,result)
    # redSample(result,img)
    # porphyinSample(img,result)
    # acneSample(img,result)
    # redSample(result,img,"right")
    # redSample(result,img,"left")
    # findForeHand_skin(img,result)

    # imgd=drawPoints(img,result)
    # cv2.namedWindow("test1",2)
    # cv2.resizeWindow("test1",imgd.shape[0],imgd.shape[1])
    # cv2.imshow("test1",imgd)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
