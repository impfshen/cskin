#!/usr/bin/env python
from __future__ import print_function

import sys
import time
import os
import yaml
from Queue import Queue
import threading
from camera import sysCamera
from ipad import sysIPad
import sys
sys.path.append(r'/home/cskin')
from src_terminal.fileadjust import Adjust
from src_terminal.filecopy import Copy


def clearQueue(q):
    # locks and clears a queue
    with q.mutex:
        q.queue.clear()


def checkfile(filename):
    path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(path):
        print('Please provide {}'.format(filename))
        return 0
    return 1


def main():
    if not checkfile('config.yml') or not checkfile('msg.yml'):
        return 1

    with open('msg.yml', 'r') as f:
        messages = yaml.load(f)

    upload_queue = Queue()
    process_queue = Queue()
    capture_queue = Queue()

    cameras = sysCamera()
    # 2345 can be any number, we just need the ipad to also listen on this port
    iPad = sysIPad(2345)

    def upload_worker():
        # This is an undying worker that keeps trying to upload images in the image queue
        while True:
            time.sleep(0.2)
            if (upload_queue.empty()):
                # nothing to upload
                continue
            img_name = upload_queue.get()
            # usually, image name would be of format position_color
            # this message is put by the capture_worker after all images are taken
            if (img_name == messages['FINISH_MSG']):
                # send message finished to ipad
                iPad.sendMessage(messages['FINISH_MSG'])
                # next loop
                continue
            status = iPad.uploadImage(img_name)
            if (status == -1):
                print("Upload image {} failed.".format(img_name))

    u_thread = threading.Thread(target=upload_worker)
    u_thread.daemon = True  # This kills subthread when main thread dies
    u_thread.start()

    def process_worker():
        while True:
            time.sleep(0.2)
            if (process_queue.empty()):
                continue
            filename = process_queue.get()
            if (filename == messages['FINISH_MSG']):
                upload_queue.put(messages['FINISH_MSG'])
            try:
                Adjust(filename)
            except Exception:
                Copy(filename)
            upload_queue.put(filename)

    p1_thread = threading.Thread(target=process_worker)
    p1_thread.daemon = True
    p1_thread.start()
    
    p2_thread = threading.Thread(target=process_worker)
    p2_thread.daemon = True
    p2_thread.start()

    p3_thread = threading.Thread(target=process_worker)
    p3_thread.daemon = True
    p3_thread.start()
    
    p4_thread = threading.Thread(target=process_worker)
    p4_thread.daemon = True
    p4_thread.start()
    
    p5_thread = threading.Thread(target=process_worker)
    p5_thread.daemon = True
    p5_thread.start()
    
    p6_thread = threading.Thread(target=process_worker)
    p6_thread.daemon = True
    p6_thread.start()

    p7_thread = threading.Thread(target=process_worker)
    p7_thread.daemon = True
    p7_thread.start()
    
    p8_thread = threading.Thread(target=process_worker)
    p8_thread.daemon = True
    p8_thread.start()
    '''
    p9_thread = threading.Thread(target=process_worker)
    p9_thread.daemon = True
    p9_thread.start()
    
    p10_thread = threading.Thread(target=process_worker)
    p10_thread.daemon = True
    p10_thread.start()
    '''
    def capture_worker():
        while True:
            time.sleep(0.05)
            if (capture_queue.empty()):
                # nothing to capture
                continue
            s = capture_queue.get()
            if (s == messages['FINISH_MSG']):
                # This indicates the end of a capture sequence
                process_queue.put(messages['FINISH_MSG'])
                cameras.led.top_light()
                cameras.exitCameras()
                # start a new loop
                continue
            filename = cameras.capture(s)
            if (filename):
                process_queue.put(filename)

    c_thread = threading.Thread(target=capture_worker)
    c_thread.daemon = True
    c_thread.start()

    cameras.printStatus()

    while True:
        while not iPad.isConnected():
            # While not connected, wait for ipad connection
            iPad.connect()
            time.sleep(1)

        msg = iPad.getMsg()
        # msg = [[::Real Message here::]]
        if (msg == messages['CAPTURE_MSG']):
            for s in cameras.getSequence():
                capture_queue.put(s)
            capture_queue.put(messages['FINISH_MSG'])
        elif (msg == messages['CANCEL_MSG']):
            clearQueue(upload_queue)
            clearQueue(process_queue)
            clearQueue(capture_queue)
            cameras.exitCameras()
            cameras.led.top_light()
        elif (msg == ""):
            # Empty message, clear queues
            clearQueue(upload_queue)
            clearQueue(process_queue)
            clearQueue(capture_queue)
            cameras.exitCameras()
            # reset connection
            iPad.setDisconnectByRemote()
        elif (msg == messages['SAVE_CONFIG_MSG']):
            cameras.writeConfig()
        else:
            m = msg.split(":")
            if (m[0] == messages['UPDATE_LED_MSG']):
                # update_led:left:green:00 00 13 14 00 ff 00 aa bb cc dd
                cameras.updateLedCommand(m[1], m[2], m[3])
            elif (m[0] == messages['CONNECT_CAMERA_MSG']):
                # connect_camera:left
                cameras.connectFirstUnkonwnCamera(m[1])
            else:
                print("Received unknown message {}".format(msg))
                iPad.setDisconnectByRemote()


if __name__ == "__main__":
    sys.exit(main())
