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

import shutil
import json
import argparse
import os
import tqdm
import numpy as np
import cv2
import MxpiDataType_pb2 as MxpiDataType
from PIL import Image
from utils import preprocess
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
from StreamManagerApi import StreamManagerApi, MxProtobufIn, InProtobufVector, StringVector

if __name__ == '__main__':
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    pipeline = {
        "Retinaface": {
            "stream_config": {
                "deviceId": "3"
            },
            "appsrc0": {
                "props": {
                    "blocksize": "409600"
                },
                "factory": "appsrc",
                "next": "mxpi_tensorinfer0"
            },
            "mxpi_tensorinfer0": {
                "props": {
                    "singleBatchInfer": "1",
                    "dataSource": "appsrc0",
                    "modelPath": "./model/newRetinaface.om"
                },
                "factory": "mxpi_tensorinfer",
                "next": "mxpi_objectpostprocessor0"
            },
            "mxpi_objectpostprocessor0": {
            "props": {
                "dataSource": "mxpi_tensorinfer0",
                "postProcessConfigPath": "./config/face_Retina.cfg",
                "postProcessLibPath": "libtotalyunetpostprocess.so"
            },
            "factory": "mxpi_objectpostprocessor",
            "next": "appsink0"
            },
            "appsink0": {
                "props": {
                    "blocksize": "409600"
                },
                "factory": "appsink"
            }
        }
    }
    pipelineStr = json.dumps(pipeline).encode()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    tensor_data , return_img = preprocess("./test.jpg")
    tensor = tensor_data[None, :]

    STREAMNAME = b"Retinaface"
    INPLUGINID = 0
    visionList = MxpiDataType.MxpiVisionList()
    visionVec = visionList.visionVec.add()
    visionInfo = visionVec.visionInfo

    visionInfo.width = 1000
    visionInfo.height = 1000
    visionInfo.widthAligned = 1000
    visionInfo.heightAligned = 1000
    visionData = visionVec.visionData
    visionData.dataStr = tensor.tobytes()
    visionData.deviceId = 0
    visionData.memType = 0
    visionData.dataSize = len(tensor)
    
    KEY_VALUE = b"appsrc0"
    protobufVec = InProtobufVector()
    
    protobuf = MxProtobufIn()
    protobuf.key = KEY_VALUE
    protobuf.type = b"MxTools.MxpiVisionList"
    protobuf.protobuf = visionList.SerializeToString()
    protobufVec.push_back(protobuf)
 
    uniqueId = streamManagerApi.SendProtobuf(STREAMNAME, INPLUGINID, protobufVec)

    keys = [b'mxpi_objectpostprocessor0']
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)
    infer_result = streamManagerApi.GetProtobuf(STREAMNAME, 0, keyVec)
    if infer_result.size() == 0:
        print("infer_result is null")
        exit()

    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
        exit()
    objectList = MxpiDataType.MxpiObjectList()
    objectList.ParseFromString(infer_result[0].messageBuf)

    img = cv2.imread('test.jpg')
    result = objectList.objectVec
    resize , left, top, right, bottom = return_img
    for x in result:
        new_x0 = max(int((x.x0-left)/resize), 0)
        new_x1 = max(int((x.x1-left)/resize), 0)
        new_y0 = max(int((x.y0-top)/resize), 0)
        new_y1 = max(int((x.y1-top)/resize), 0)

        confidence = x.classVec[0].confidence
        cv2.rectangle(img, (new_x0, new_y0), (new_x1, new_y1), (255, 0, 0), 2)

    cv2.imwrite("./result.jpg", img)
    streamManagerApi.DestroyAllStreams()
