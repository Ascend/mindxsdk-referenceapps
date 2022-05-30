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




import os
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
    with open("./pipeline/Yunet.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = STREAM_MANAGER_API.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # get result from stream and write
    STREAM_NAME = b'Yunet'
    IN_PLUGIN_ID = 0
    FRAME_COUNT = 0

    FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    MODES = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open('./result.264', FLAGS, MODES), 'w') as fp:
        while FRAME_COUNT <= 100:
            FRAME_COUNT += 1
            inferResult = STREAM_MANAGER_API.GetResult(STREAM_NAME, IN_PLUGIN_ID)
            fp.write(inferResult.data)
    
# destroy streams
STREAM_MANAGER_API.DestroyAllStreams()



