#!/usr/bin/env python
# coding=utf-8

"""
Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""



import sys
import re
import json
import os
import stat
import random
import signal
import datetime
import threading
import cv2
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from PIL import Image
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector


def sigint_handler(signum, frame):
    signum = signum
    frame = frame
    global ISSIGINTUP
    ISSIGINTUP = True
    print("catched interrupt signal")

def stop_thread():
    global OVER
    OVER = True

signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGHUP, sigint_handler)
signal.signal(signal.SIGTERM, sigint_handler)
ISSIGINTUP = False
OVER = False

if __name__ == '__main__':
    
    # init stream manager
    STREAM_MANAGER_API = StreamManagerApi()
    ret = STREAM_MANAGER_API.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./pipeline/InferTest.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = STREAM_MANAGER_API.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # get result from stream and write
    STREAM_NAME = b'Yunet'
    IN_PLUGIN_ID = 0
    FRAME_COUNT = 0

    def time_func():
        time_step = 0
        time_count = 0
        begin_time = datetime.datetime.now()
        one_step = 10

        while True:
            cur_time = (datetime.datetime.now() - begin_time).total_seconds()
            if cur_time >= (time_step + one_step):
                time_step = time_step + one_step
                print("10秒平均帧率:", (FRAME_COUNT - time_count) * 1.0 / one_step)
                time_count = FRAME_COUNT
            if ISSIGINTUP or OVER:
                print("Exit")
                break

    t = threading.Thread(target=time_func, args=())
    t.start()
    FLAGS = os.O_WRONLY | os.O_CREAT
    MODES = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open('./result.264', FLAGS, MODES), 'wb') as fp:
        while FRAME_COUNT <= 100:
            FRAME_COUNT += 1
            inferResult = STREAM_MANAGER_API.GetResult(STREAM_NAME, IN_PLUGIN_ID)
            fp.write(inferResult.data)
            # print(inferResult.data)
            if ISSIGINTUP:
                print("Exit")
                break

    # destroy streams
    STREAM_MANAGER_API.DestroyAllStreams()
    stop_thread()