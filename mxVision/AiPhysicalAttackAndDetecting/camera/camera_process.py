#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
Description: Input source process.
Author: MindX SDK
Create: 2023
History: NA
"""

import os 
import signal
import configparser
from configparser import ConfigParser
import cv2


def get_video_source(path="config/config.ini") -> str:
    if not os.path.exists(path):
        raise FileExistsError("config/config.ini not found, please check.")
    config = ConfigParser()
    config.read(path)
    if "camera" not in config and "video" not in config:
        raise configparser.NoSectionError("please set camera or video info.")
    if "camera" in config:
        if "rtsp" not in config["camera"]:
            raise configparser.NoOptionError("rtsp", "camera")
        video_source = config["camera"]["rtsp"]
        if video_source:
            return video_source
    if "video" in config:
        if "video_path" not in config["video"]:
            raise configparser.NoOptionError("video_path", "video")
    
    video_source = config["video"]["video_path"]
    if not os.path.exists(video_source):
        raise FileNotFoundError("video not exist, please check.")
    return video_source


class Camera:
    def __init__(self):
        self.video_source = get_video_source()
        self.camera = cv2.VideoCapture(self.video_source)
 
    def __del__(self):
        self.camera.release()

    def get_frame_from_camera(self, my_queue):
        count = 0
        while self.camera.isOpened():
            ok, frame = self.camera.read()
            if ok:
                count = 0
                if my_queue.full():
                    my_queue.get()
                my_queue.put(frame)
            else:
                count += 1
                if count > 150:
                    print("连续150次未能获取到输入，退出。")
                    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


def get_frame(my_queue):
    my_cam = Camera()
    my_cam.get_frame_from_camera(my_queue)
