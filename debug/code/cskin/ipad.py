#!/usr/bin/env python
from __future__ import print_function
import usbmux
import socket
import time
import os
import shutil
import sys
sys.path.append(r'/home/cskin')
from src_terminal.filecopy import Copy
from src_terminal.fileadjust import Adjust
from src_terminal.filepath import colorPath,inPath,outPath

class sysIPad(object):
    """docstring for sysIPad."""

    def __init__(self, port):
        super(sysIPad, self).__init__()
        self.mux = None
        self.connected = False
        self.psock = None
        self.recvMsg = ""
        self.port = port
        self.send_msg_prefix = "msg://"
        self.send_file_prefix = "file://"
        self.prepacket = self.send_msg_prefix
        self.afterpacket = "::]]"
        self.send_size = 10240
        self.status = None

    def connect(self):
        try:
            self.mux = usbmux.USBMux()
            if not self.mux.devices:
                self.mux.process(0.1)
        except socket.error as ex:
            if self.status != "sockrefused":
                print("usbmuxd service failed. No problem.")
                self.status = "sockrefused"
            return

        if not self.mux.devices:
            if self.status != "waiting":
                print("Waiting for iPad...")
                self.status = "waiting"
            return
        try:
            self.psock = self.mux.connect(self.mux.devices[0], self.port)
            self.connected = True
            self.status = "connected"
            print("Connected!")
            return
        except Exception as er:
            if self.status != "sockunavaliable":
                print("Waiting for Capture...")
                self.status = "sockunavaliable"
        return

    def setDisconnectByRemote(self):
        self.connected = False
        self.mux = None
        self.psock.close()
        return

    def getMsg(self):
        recvMsg = self.psock.recv(64)
        # If prepacket or afterpacket is missing,
        # this is not a valid message
        if (self.prepacket not in recvMsg) or (self.afterpacket not in recvMsg):
            self.recvMsg = ""
            return self.recvMsg

        # finds the first index where prepacket is shown
        pre_index = recvMsg.find(self.prepacket)
        # this finds the afterpacket from substring after pre_index till end
        sux_index = recvMsg.find(self.afterpacket, pre_index)
        # Now we know the index,
        # we can get the message in between prepacket and afterpacket
        self.recvMsg = recvMsg[len(self.prepacket) + pre_index:sux_index]
        return self.recvMsg

    def isConnected(self):
        return self.connected

    def sendImgAfterpacket(self):
        self.psock.send(self.afterpacket)

    def buildPacket(self, to_build, size):
        if (len(to_build) > size):
            raise Exception("Oversized packet")
        return to_build + " " * (size - len(to_build))

    def sendImgDetails(self, image_name):
        first = image_name.find("_")
        # removed .jpg at the end of image_name
        to_build = self.send_file_prefix + image_name[first + 1:
                                                      len(image_name) - 4]
        # The first 20 bytes contain image name!
        # This a protocol we agreed with the other side!
        to_send = self.buildPacket(to_build, 20)
        self.psock.send(to_send)

    def sendMessage(self, msg):
        to_send = self.prepacket + msg + self.afterpacket
        self.psock.send(to_send)

    def uploadImage(self, image_name):
        if image_name == None:
            return

        #Adjust(image_name)

        target = os.path.join(os.getcwd(), 'output', image_name)

        img = open(target, 'rb')
        l = img.read(self.send_size)
        try:
            self.sendImgDetails(image_name)
            while (l):
                self.psock.send(l)
                l = img.read(self.send_size)
            self.sendImgAfterpacket()
        except socket.error:
            img.close()
            return -1

        img.close()
        print("upload {} complete".format(image_name))

        #os.remove(inPath+'/'+image_name)
        #os.remove(outPath+'/'+image_name)

        return 0
