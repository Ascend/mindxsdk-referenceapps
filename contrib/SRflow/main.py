#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import json
import os
import sys
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxProtobufIn, InProtobufVector, StringVector
import cv2
import numpy as np
from utils import preprocess , postprocess , valid

if __name__ == '__main__':
    
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    pipeline = {
        "superResolution": {
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
                    "dataSource": "appsrc0",
                    "modelPath": "./model/srflow_df2k_x8_bs1.om"
                },
                "factory": "mxpi_tensorinfer",
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

    tensor_data , origin_size = preprocess("./image/test.png")
    tensor = tensor_data[None, :]

    STREAMNAME = b'superResolution'
    INPLUGINID = 0
    visionList = MxpiDataType.MxpiVisionList()
    visionVec = visionList.visionVec.add()
    visionInfo = visionVec.visionInfo
    # The standard input size of srflow model is 256
    visionInfo.width = 256
    visionInfo.height = 256
    visionInfo.widthAligned = 256
    visionInfo.heightAligned = 256
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

    # get plugin output data
    KEY_VALUE = b"mxpi_tensorinfer0"
    keyVec = StringVector()
    keyVec.push_back(KEY_VALUE)
    inferResult = streamManagerApi.GetProtobuf(STREAMNAME, 0, keyVec)
    if inferResult.size() == 0:
        print("inferResult is null")
        exit()
    if inferResult[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
            inferResult[0].errorCode, inferResult[0].messageName.decode()))
        exit()

    # get the infer result
    infer_list = MxpiDataType.MxpiTensorPackageList()
    infer_list.ParseFromString(inferResult[0].messageBuf)
    infer_data = infer_list.tensorPackageVec[0].tensorVec[0].dataStr
    output = np.frombuffer(infer_data, dtype=np.float32)

    hr = cv2.imread("./image/test_hr.png")
    img = postprocess(output , hr.shape)
    cv2.imwrite("result.jpg", img)
    psnr_val = valid(img , hr)
    print("Infer finished.")
    streamManagerApi.DestroyAllStreams()