#!/usr/bin/env python
from __future__ import print_function

import logging
import os
import sys
import yaml
from led import LED
from os.path import join, dirname
import gphoto2 as gp

# Global variables for this file
toConfig = ["aperture", "shutterspeed", "iso", "whitebalance"]
positions = ['left', 'right', 'middle']


class sysCamera(object):
    """docstring for sysCamera."""

    def __init__(self):
        super(sysCamera, self).__init__()

        # read yaml from config.yml
        with open('config.yml', 'r') as f:
            self.settings = yaml.load(f)

        # logging for libgphoto2
        logging.basicConfig(
            format='%(levelname)s: %(name)s: %(message)s', level=logging.ERROR)
        gp.check_result(gp.use_python_logging())

        self.cameras = {}
        # Detect all cameras and load them into self.cameras
        self.detectCameras()
        # make every camera not silent
        for pos in self.cameras:
            camera, context = self.cameras[pos]
            config = camera.get_config(context)
            self.set_config(config, 'drivemode', 'Single')
            camera.set_config(config, context)
            # free the camera
            camera.exit(context)

        led_settings = self.settings['led']
        self.led = None
        if 'port' in led_settings:
            self.led = LED(led_settings['port'], led_settings['commands'])
        else:
            print("LED port not given in config.yml")

    def updateLedCommand(self, pos, color, command):
        # this sets the actual led setting
        self.led.set_command(pos, color, command)
        # this keeps track of the settings here
        self.settings['led'][pos + "_" + color] = command
        return

    def exitCameras(self):
        for pos in self.cameras:
            camera, context = self.cameras[pos]
            camera.exit(context)
        return

    def detectCameras(self):
        '''
        This function autodetects all connected cameras and initialize them into self.cameras.
        If serial is known, this will write to corresponding position, otherwise, this will write with an increasing number (0, 1, 2...)
        '''
        # Assume self.settings is already updated

        # reset self.cameras
        self.cameras = {}

        (camera_info_list, port_info_list) = self.autodetect()
        if not camera_info_list:
            print('No camera detected!')
            return
        count = 0
        # c_info is of format (camera_type, camera_usb_port)
        for c_info in camera_info_list:
            if (c_info[0] != "Canon EOS 100D"):
                continue
            context = gp.gp_context_new()
            try:
                # we initialize the camera
                camera = self.camera_init(c_info[1], port_info_list, context)
            except gp.GPhoto2Error as ex:
                print('Initialization of camera at {} failed'.format(c_info[1]))
                print('Error: {}'.format(ex))
                # Try to initialize next camera
                continue
            # So here we have successfully initialized the camera
            serial = self.getCameraSerial(camera, context)
            pos = self.matchSerial(serial)
            if (pos):
                self.cameras[pos] = (camera, context)
                print("Identified Camera: {} {}".format(pos, serial))
            else:
                self.cameras[count] = (camera, context)
                print("Unidentified Camera {} : {}".format(count, serial))
                count += 1
        return

    def autodetect(self):
        length, camera_info_list = gp.gp_camera_autodetect(None)
        if not camera_info_list:
            return (None, None)
        port_info_list = gp.PortInfoList()
        port_info_list.load()
        return (camera_info_list, port_info_list)

    def matchSerial(self, serial):
        factory_serials = self.settings['camera_serial']
        pos = None
        if serial == factory_serials['left']:
            pos = 'left'
        elif serial == factory_serials['right']:
            pos = 'right'
        elif serial == factory_serials['middle']:
            pos = 'middle'
        return pos

    def getCameraSerial(self, camera, context):
        '''
        Given a camera and its context, get its serialnumber
        '''
        config = gp.check_result(gp.gp_camera_get_config(camera, context))
        serial = self.get_config(config, "serialnumber")
        # free the camera
        gp.check_result(gp.gp_camera_exit(camera, context))
        return serial

    def printStatus(self):
        self.reloadConfig()
        toRedetect = False
        for pos in positions:
            serial = self.settings['camera_serial'][pos]
            if (serial):
                print("Config has Camera: {} {}".format(pos, serial))
            else:
                print("Config misses Camera: {}".format(pos))
        for pos in self.cameras:
            if pos in positions:
                print("Identified Camera: {}".format(pos))
            else:
                toRedetect = True
                print("Unkonwn camera : {}".format(pos))
        if toRedetect:
            self.autodetect()

    def connectFirstUnkonwnCamera(self, desired_pos):
        to_pop = None
        for pos in self.cameras:
            if pos in positions:
                # Has position
                continue
            self.cameras[desired_pos] = self.cameras[pos]
            to_pop = pos
            camera, context = self.cameras[desired_pos]
            serial = self.getCameraSerial(camera, context)
            self.connectCamera(desired_pos, serial)
            break
        self.cameras.pop(to_pop, None)

    def connectCamera(self, pos, serial):
        self.settings['camera_serial'][pos] = serial
        self.writeConfig()
        print('Connected {} with serial: {}'.format(pos, serial))

    def resetCameras(self):
        for pos in positions:
            self.resetCamerasPos(pos)

    def resetCamerasPos(self, pos):
        if pos not in positions:
            print('Unknown position given: {}'.format(pos))
            return
        self.settings['camera_serial'][pos] = ''
        self.writeConfig()
        print('Camera {} reset'.format(pos))

    def writeConfig(self):
        new_setting = yaml.dump(self.settings)
        with open('config.yml', 'w') as f:
            f.write(new_setting)
        return

    def reloadConfig(self):
        # Reload config
        with open('config.yml', 'r') as f:
            self.settings = yaml.load(f)

    def apply_settings(self, setting, config):
        for cfg in toConfig:
            self.set_config(config, cfg, setting[cfg])
        return

    # ########################
    # Camera actions
    # ########################
    def camera_init(self, addr, pil, context):
        """
        :param addr: the camera address
        :param pil: port infor list, if not given, will try to get a port info
        """
        if (pil == None):
            # Get port info list if not given
            pil = gp.PortInfoList()
            pil.load()
        # Set up this given camera
        camera = gp.Camera()
        idx = pil.lookup_path(addr)
        camera.set_port_info(pil[idx])
        camera.init(context)
        return camera

    def get_avaliable_config(self, config, name):
        """
        find the camera config item values
        """
        item = gp.check_result(gp.gp_widget_get_child_by_name(config, name))
        count = gp.check_result(gp.gp_widget_count_choices(item))
        # make sure value >= 0 and value < count
        print('Available {0} Settings'.format(name.upper()))
        print('=======')
        for choice in range(count):
            value = gp.check_result(gp.gp_widget_get_choice(item, choice))
            print('{0}: {1}'.format(choice, value))
        print('\n')

    def get_config(self, config, name):
        """
        get the current config
        """
        item = gp.check_result(gp.gp_widget_get_child_by_name(config, name))
        return gp.check_result(gp.gp_widget_get_value(item))

    def set_config(self, config, name, value):
        """
        set the specific configuration of a config with value = value
        """
        item = gp.check_result(gp.gp_widget_get_child_by_name(config, name))
        gp.check_result(gp.gp_widget_set_value(item, value))

    def display_config(self, config):
        """
        print camera configs
        """
        print("=" * 10)
        for c in config:
            print("{0} : {1}".format(c, config[c]))

    def getSequence(self):
        return self.settings['sequence']

    def capture(self, settings):
        """
        capture an image and store at local
        """
        if (not self.cameras):
            # No cameras connected
            return

        pos = settings['pos']
        color = settings['color']
        filename = 'img_{}_{}.jpg'.format(pos, color)
        print('Capturing {}_{}'.format(pos, color))

        targetDir = os.path.join(os.getcwd(), 'input')
        if not os.path.exists(targetDir):
            os.mkdir(targetDir)
        target = os.path.join(os.getcwd(), 'input', filename)

        try:
            camera, context = self.cameras[pos]
        except KeyError:
            print("Camera {} not found. Capture request ignored".format(pos))
            return None


        try:
            # set config
            config = camera.get_config(context)
            self.apply_settings(settings, config)
            camera.set_config(config, context)
            # turn light up
            if self.led:
                self.led.light_up(pos, color)
            # capture
            file_path = gp.check_result(
                gp.gp_camera_capture(camera, gp.GP_CAPTURE_IMAGE, context))
            # turn off light
            if self.led:
                self.led.turn_off()
            # save file
            camera_file = gp.check_result(
                gp.gp_camera_file_get(camera, file_path.folder, file_path.name,
                                      gp.GP_FILE_TYPE_NORMAL, context))
            gp.check_result(gp.gp_file_save(camera_file, target))
        except gp.GPhoto2Error as ex:
            if self.led:
                self.led.turn_off()
            filename = None
            camera.exit(context)
            if ex.code != -108:
                print(ex)

        return filename

    def testCapture(self, filename, camera, context, turn_off):
        """
        capture an image and store at local
        """
        try:
            file_path = camera.capture(gp.GP_CAPTURE_IMAGE, context)
        except gp.GPhoto2Error as ex:
            print("Gphoto error : {}".format(ex))
            return None

        if turn_off:
            turn_off()

        # save to local
        targetDir = os.path.join(os.getcwd(), 'input')
        if not os.path.exists(targetDir):
            os.mkdir(targetDir)
        target = os.path.join(os.getcwd(), 'input', filename)
        print('Copying image to', target)
        camera_file = gp.check_result(
            gp.gp_camera_file_get(camera, file_path.folder, file_path.name,
                                  gp.GP_FILE_TYPE_NORMAL, context))
        gp.check_result(gp.gp_file_save(camera_file, target))
        # Free the camera
        gp.check_result(gp.gp_camera_exit(camera, context))
        return target

    def camera_test(self):
        # Try 10 ten times to detect cameras
        count = 0
        while (count < 10):
            if (not self.cameras):
                self.detectCameras()
                count += 1
            else:
                break
        # check if each camera can take a picture
        for pos in self.cameras:
            camera, context = self.cameras[pos]
            img_name = 'test_{}.jpg'.format(pos)
            self.testCapture(img_name, camera, context, None)


if __name__ == "__main__":
    cameras = sysCamera()
    cameras.camera_test()
