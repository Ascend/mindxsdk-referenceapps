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


if __name__ == '__main__':
    # init stream manager
    STREAM_MANAGER_API = StreamManagerApi()
    ret = STREAM_MANAGER_API.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file

    with open("./pipeline/second.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = STREAM_MANAGER_API.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    IMAGE_NAME = './test.jpg'

    dataInput = MxDataInput()
    with open(IMAGE_NAME, 'rb') as f:
        dataInput.data = f.read()

    # Inputs data to a specified stream based on STREAM_NAME.
    STREAM_NAME = b'RefineDet'
    IN_PLUGIN_ID = 0
    uniqueId = STREAM_MANAGER_API.SendData(STREAM_NAME, IN_PLUGIN_ID, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()
 
    inferResult = STREAM_MANAGER_API.GetResult(STREAM_NAME, IN_PLUGIN_ID)

    FLAGS = os.O_WRONLY | os.O_CREAT
    MODES = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open('./out.jpg', FLAGS, MODES), 'wb') as f:
        f.write(inferResult.data)

    # destroy streams
    STREAM_MANAGER_API.DestroyAllStreams()