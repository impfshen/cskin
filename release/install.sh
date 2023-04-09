#!/bin/bash

#move brightness.tar.gz from somewhere to /home/cskin
#un-compress brightness.tar.gz, get install.sh,code.tar.gz

#backup
sudo cp -r ~/cskin-firmware/cskin ~/cskin-firmware/cskin_bak

#un-compress
sudo tar -xzvf code.tar.gz
#sudo tar -xzvf cskin-firmware.tar.gz

#delete main.py,ipad.py,camera.py
sudo rm -f ~/cskin-firmware/cskin/main.py
sudo rm -f ~/cskin-firmware/cskin/ipad.py
sudo rm -f ~/cskin-firmware/cskin/camera.py
sudo rm -f ~/cskin-firmware/cskin/error.log

#replace files above
sudo cp ~/code/cskin/main.py ~/cskin-firmware/cskin/main.py
sudo cp ~/code/cskin/ipad.py ~/cskin-firmware/cskin/ipad.py
sudo cp ~/code/cskin/camera.py ~/cskin-firmware/cskin/camera.py
sudo cp ~/code/cskin/error.log ~/cskin-firmware/cskin/error.log

#copy src_terminal
sudo rm -rf ~/src_terminal
sudo cp -r ~/code/src_terminal ~/src_terminal

#copy AIData
sudo rm -rf ~/cskin-firmware/cskin/input
sudo rm -rf ~/cskin-firmware/cskin/output
sudo rm -rf ~/cskin-firmware/cskin/copy
sudo rm -rf ~/cskin-firmware/cskin/color
sudo rm -rf ~/cskin-firmware/cskin/standard
sudo rm -rf ~/cskin-firmware/cskin/location
sudo rm -rf ~/cskin-firmware/cskin/test
sudo cp -r ~/code/AIData/input ~/cskin-firmware/cskin/input
sudo cp -r ~/code/AIData/output ~/cskin-firmware/cskin/output
sudo cp -r ~/code/AIData/copy ~/cskin-firmware/cskin/copy
sudo cp -r ~/code/AIData/color ~/cskin-firmware/cskin/color
sudo cp -r ~/code/AIData/standard ~/cskin-firmware/cskin/standard
sudo cp -r ~/code/AIData/location ~/cskin-firmware/cskin/location
sudo cp -r ~/code/AIData/test ~/cskin-firmware/cskin/test

#install python packdage
sudo apt-get -y install python-numpy
sudo apt-get -y install python-scipy
sudo apt-get -y install python-matplotlib
sudo apt-get -y install python-skimage
sudo apt-get -y install python-opencv
sudo apt-get -y install python-yaml
sudo apt-get -y install python-serial
sudo apt-get -y install python-gphoto2

#set auto_startup
sudo systemctl stop cskin
sudo rm -f /etc/systemd/system/cskin.service
sudo cp ~/code/cskin.service /etc/systemd/system/cskin.service
sudo systemctl daemon-reload
sudo systemctl start cskin

#set file priority
sudo chmod 777 ~/src_terminal
sudo chmod 777 ~/src_terminal/adjustcolor.py
sudo chmod 777 ~/src_terminal/adjust_color2.py
sudo chmod 777 ~/src_terminal/color.py
sudo chmod 777 ~/src_terminal/colorList.py
sudo chmod 777 ~/src_terminal/detect_color.py
sudo chmod 777 ~/src_terminal/fileadjust.py
sudo chmod 777 ~/src_terminal/filecopy.py
sudo chmod 777 ~/src_terminal/filepath.py
sudo chmod 777 ~/src_terminal/__init__.py
sudo chmod 777 ~/src_terminal/ioexception.py
sudo chmod 777 ~/src_terminal/seetaUtil.py
sudo chmod 777 ~/src_terminal/rotate.py
sudo chmod 777 ~/src_terminal/test.py
sudo chmod 777 ~/cskin-firmware/cskin/main.py
sudo chmod 777 ~/cskin-firmware/cskin/ipad.py
sudo chmod 777 ~/cskin-firmware/cskin/camera.py
sudo chmod 777 ~/cskin-firmware/cskin/error.log
sudo chmod 777 ~/cskin-firmware/cskin/input
sudo chmod 777 ~/cskin-firmware/cskin/output
sudo chmod 777 ~/cskin-firmware/cskin/copy
sudo chmod 777 ~/cskin-firmware/cskin/color
sudo chmod 777 ~/cskin-firmware/cskin/standard
sudo chmod 777 ~/cskin-firmware/cskin/location
sudo chmod 777 ~/cskin-firmware/cskin/test
sudo chmod 777 ~/cskin-firmware/cskin/standard/white_standard.txt
sudo chmod 777 ~/cskin-firmware/cskin/standard/upw_standard.txt
sudo chmod 777 ~/cskin-firmware/cskin/standard/uv_standard.txt
sudo chmod 777 ~/cskin-firmware/cskin/standard/red_standard.txt
sudo chmod 777 ~/cskin-firmware/cskin/standard/blue_standard.txt
sudo chmod 777 ~/cskin-firmware/cskin/standard/green_standard.txt
sudo chmod 777 ~/cskin-firmware/cskin/standard/brown_standard.txt
sudo chmod 777 ~/cskin-firmware/cskin/location/ColorPointLeft.txt
sudo chmod 777 ~/cskin-firmware/cskin/location/ColorPointMiddle.txt
sudo chmod 777 ~/cskin-firmware/cskin/location/ColorPointRight.txt
sudo chmod 777 ~/cskin-firmware/cskin/test/mode.txt
sudo chmod 777 ~/cskin-firmware/cskin/test/img_left_blue.txt
sudo chmod 777 ~/cskin-firmware/cskin/test/img_left_green.txt
sudo chmod 777 ~/cskin-firmware/cskin/test/img_left_white.txt
sudo chmod 777 ~/cskin-firmware/cskin/test/img_right_blue.txt
sudo chmod 777 ~/cskin-firmware/cskin/test/img_right_green.txt
sudo chmod 777 ~/cskin-firmware/cskin/test/img_right_white.txt
sudo chmod 777 ~/cskin-firmware/cskin/test/img_middle_blue.txt
sudo chmod 777 ~/cskin-firmware/cskin/test/img_middle_green.txt
sudo chmod 777 ~/cskin-firmware/cskin/test/img_middle_white.txt
sudo chmod 777 ~/cskin-firmware/cskin/test/img_middle_upw.txt
sudo chmod 777 ~/cskin-firmware/cskin/test/img_middle_uv.txt



sudo tar -xzvf tool.tar.gz
sudo chmod 777 ~/tool/standard_color.py
sudo chmod 777 ~/tool/locate.py
sudo chmod 777 ~/tool/filepath.py
sudo chmod 777 ~/tool/colorList.py
sudo chmod 777 ~/tool/__init__.py
sudo chmod 777 ~/tool/color
sudo chmod 777 ~/standardImg
sudo chmod 777 ~/record
sudo chmod 777 ~/standardImg/blue.jpg
sudo chmod 777 ~/standardImg/green.jpg
sudo chmod 777 ~/standardImg/red.jpg
sudo chmod 777 ~/standardImg/white.jpg
sudo chmod 777 ~/standardImg/upw.jpg
sudo chmod 777 ~/standardImg/uv.jpg
sudo chmod 777 ~/standardImg/brown.jpg
sudo chmod 777 ~/record/img_left_white.jpg
sudo chmod 777 ~/record/img_middle_white.jpg
sudo chmod 777 ~/record/img_right_white.jpg


#standard image data
sudo python2 ~/tool/standard_color.py


#locate color point
sudo python2 ~/tool/locate.py


#finish and reboot
sudo rm -rf ~/code
sudo rm -rf ~/tool
sudo rm -f ~/tool.tar.gz
sudo rm -f ~/code.tar.gz
sudo rm -f ~/install.sh
echo ===================================================
echo Install Success. Please run sudo reboot to Reboot
echo ===================================================
