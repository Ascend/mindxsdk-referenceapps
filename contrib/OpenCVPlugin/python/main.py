#!/usr/bin/env python
# coding=utf-8

"""
Copyright(C) Huawei Technologies Co.,Ltd. 2012-2021 All rights reserved.

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

import json
import os
import sys
import cv2
import numpy as np

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

if __name__ == '__main__':
    streamManagerApi = StreamManagerApi()

    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()


    
    with open("./test.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    
    dataInput = MxDataInput()
    if os.path.exists('person.jpg') != 1:
        print("The test image does not exist.")
        sys.exit(0)
    with open("person.jpg", 'rb') as f:
        dataInput.data = f.read()

    STREAM_NAME = b'detection'
    INPLUGIN_ID = 0
  
    uniqueId = streamManagerApi.SendData(STREAM_NAME, INPLUGIN_ID, dataInput)

    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    keys = [b"mxpi_OpenCVPlugin"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)

    infer_result = streamManagerApi.GetProtobuf(STREAM_NAME, 0, keyVec)

    if infer_result.size() == 0:
        print("infer_result is null")
        exit()

    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" % (
            infer_result[0].errorCode))
        exit()

    YUV_BYTES_NU = 3
    YUV_BYTES_DE = 2
    visionList = MxpiDataType.MxpiVisionList()
    visionList.ParseFromString(infer_result[0].messageBuf)
    vision_data = visionList.visionVec[0].visionData.dataStr
    visionInfo = visionList.visionVec[0].visionInfo
    #img_yuv = np.frombuffer(vision_data, np.float32)
    img_yuv = np.frombuffer(vision_data, np.uint8) 
    #img_bgr = img_yuv.reshape(visionInfo.heightAligned, visionInfo.widthAligned,3)
    #RGB
    #img = cv2.cvtColor(img_bgr, getattr(cv2, "COLOR_BGR2RGB"))
    #BGR
    #img = cv2.cvtColor(img_bgr,getattr(cv2, "COLOR_BGR2BGRA"))

    #dvpp
    img_bgr = img_yuv.reshape(visionInfo.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, visionInfo.widthAligned)
    img = cv2.cvtColor(img_bgr, getattr(cv2, "COLOR_YUV2BGR_NV12"))

    cv2.imwrite("./result.jpg", img)
    # destroy streams
    streamManagerApi.DestroyAllStreams()
