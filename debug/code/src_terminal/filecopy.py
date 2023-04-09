import os
import shutil
import time
from ioexception import IOException
from filepath import inPath,copyPath
#from rotate import direction_correct

def Copy(filename):
    InitFileList=os.listdir(inPath)
    CopyFileList=os.listdir(copyPath)
    flag=False
    for i in InitFileList:
        if(i==filename):
            if(i in CopyFileList):
                print('File already exists in copyPath')
            else:
                #direction_correct(inPath+'/'+i)
                shutil.copy(inPath+'/'+i,copyPath+'/'+i)
                print('copy {}'.format(i))
            flag=True
            break
    if(not flag):
        print('File not exists in inPath')
