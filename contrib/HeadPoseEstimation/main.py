#!/usr/bin/env python3
# -*- coding:utf-8 -*-

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

import os

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

if __name__ == '__main__':
    # Create and initialize a new StreamManager object
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # Read and format pipeline file
    with open("./pipeline/recognition.pipeline", 'rb') as f:
        pipeline = f.read().replace(b'\r', b'').replace(b'\n', b'')
    pipelineStr = pipeline

    # Create Stream
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Create Input Object
    dataInput = MxDataInput()
    if os.path.exists('test.jpg') != 1:
        print("The test image does not exist.")

    with open("test.jpg", 'rb') as f:
        dataInput.data = f.read()

    # Stream Info
    streamName = b'recognition'
    inPluginId = 0
    # Send Input Data to Stream
    uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)

    # Get the result returned by the plugins
    keys = [b"mxpi_objectpostprocessor0", b"mxpi_tensorinfer1"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)

    outPluginId = 0
    infer_result = streamManagerApi.GetProtobuf(streamName, outPluginId, keyVec)

    if infer_result.size() == 0:
        print("infer_result is null")
        exit()

    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorPlugin=%s" % (
            infer_result[0].errorCode, infer_result[0].messageName))
        exit()

    objectList = MxpiDataType.MxpiObjectList()
    objectList.ParseFromString(infer_result[0].messageBuf)
    print(objectList)

    # destroy streams
    streamManagerApi.DestroyAllStreams()
