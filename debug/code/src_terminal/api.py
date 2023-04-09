import os
import sys
sys.path.append(r'/home/cskin/')
import shutil
from filepath import colorPath
from src_terminal.filecopy import Copy
from src_terminal.fileadjust import Adjust

if __name__ == "__main__":
    FileList=os.listdir('/home/cskin/cskin-firmware/cskin/input')
    for filename in FileList:
        print(filename)
        Adjust(filename)
        '''
        try:
            Adjust(filename)
        except:
            shutil.rmtree(colorPath+'/'+filename.rstrip('.jpg'))
            break
        '''
        

    
