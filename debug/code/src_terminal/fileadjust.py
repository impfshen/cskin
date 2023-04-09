import os
import time
import shutil
from adjust_color2 import AdjustColor2 as AdjustColor
from test import adjust_color_test as AdjustColorTest
from filepath import inPath,outPath,colorPath,standardPath,testPath
from rotate import direction_correct

def Adjust(filename):
    InputFileList=os.listdir(inPath)
    OutputFileList=os.listdir(outPath)
    flag=False
    for file in InputFileList:
        if(file==filename):
            #os.mkdir(colorPath+'/'+filename.rstrip('.jpg'))

            filePath=inPath+'/'+filename
            resultPath=outPath+'/'+filename
            #direction_correct(filePath)
            #AdjustColor(filename,filePath,resultPath)
            setting=open(testPath+'/mode.txt')
            for line in setting:
                mode=line[:-1]
            setting.close()
            if mode == 'ultimate':
                AdjustColor(filename,filePath,resultPath)
            elif mode == 'debug':
                AdjustColorTest(filename,filePath,resultPath)
            else:
                print('invalid mode')
            #shutil.rmtree(colorPath+'/'+filename.rstrip('.jpg'))
            print("adjust {}".format(filename))
            flag=True
            break
    if(not flag):
        print('File not exists in inPath')
