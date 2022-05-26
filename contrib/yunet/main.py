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



from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
import cv2
import os
import argparse
from PIL import Image


if __name__ == '__main__':
    
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./pipeline/Yunet.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # get result from stream and write
    streamName = b'Yunet'
    inPluginId = 0
    FrameCount = 0
    with open("./result.264","wb") as fp0:
        while FrameCount <= 100:
            FrameCount += 1
            inferResult = streamManagerApi.GetResult(streamName,inPluginId)
            fp0.write(inferResult.data)

# destroy streams
streamManagerApi.DestroyAllStreams()



