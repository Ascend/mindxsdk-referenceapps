#!/usr/bin/env python
#-*-coding:utf-8-*-

"""
Styletransfer for Satellite picture to map picture
"""

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


import sys
import os
import io
import cv2
from cv2 import COLOR_RGB2BGR
import numpy as np
from PIL import Image
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector



if __name__ == '__main__':
    #  check input image

    img_path = "../sat.jpg"

    # initialize the stream manager
    stream_manager = StreamManagerApi()
    stream_state = stream_manager.InitManager()

    if stream_state != 0:
        print("Failed to init Stream manager, ret=%s" % str(stream_state))
        exit()

    # create streams by the pipeline config
    with open("../pipeline/styletransfer.pipeline", 'rb') as f:
        pipeline = f.read().replace(b'\r', b'').replace(b'\n', b'')
    pipeline_string = pipeline
    stream_state = stream_manager.CreateMultipleStreams(pipeline_string)
    if stream_state != 0:
        print("Failed to create Stream, ret=%s" % str(stream_state))
        exit()

    # prepare the input of the stream #begin
    streamName = b"styletransfer"
    inPluginId = 0
    dataInput = MxDataInput()
    with open(img_path, 'rb') as f:

        dataInput.data = f.read()
    ret = stream_manager.SendData(streamName, inPluginId, dataInput)

    if ret < 0:
        print("Failed to send data to stream")
        exit()

    keys = [b"mxpi_tensorinfer0"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)

    # Get the result from the stream

    infer = stream_manager.GetResult(streamName, b'appsink0', keyVec)
    if infer.metadataVec[0].errorCode != 0:
        print("GetResult error. errorCode=%d ,errorMsg=%s" % (
            infer.metadataVec[0].errorCode, infer.metadataVec[0].errorMsg))
        exit()

    tensorList = MxpiDataType.MxpiTensorPackageList()
    tensorList.ParseFromString(infer.metadataVec[0].serializedMetadata)
    output_res_DANet = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
    # Reshape and transpose
    result = output_res_DANet.reshape(3,256,256)
    result = result.transpose(1,2,0)
    # Reverse Normalize
    result = result*255.0

    result = cv2.cvtColor(result,COLOR_RGB2BGR)
    result = cv2.resize(result,(512, 512))

    print("___________infer_finish_____________")

    cv2.imwrite('../result/map.jpg',result)

    stream_manager.DestroyAllStreams()



