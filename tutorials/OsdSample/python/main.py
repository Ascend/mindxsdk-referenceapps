#!/usr/bin/env python
# coding=utf-8

"""
Copyright 2020 Huawei Technologies Co., Ltd

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
import time
from StreamManagerApi import *
import MxpiOSDType_pb2 as MxpiOSDType
from google.protobuf.json_format import *

if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("../pipeline/SampleOsd.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    dataInput = MxDataInput()
    with open("test.jpg", 'rb') as f:
        dataInput.data = f.read()

    # Send image.
    streamName = b'encoder'
    inPluginId = 0
    ret = streamManagerApi.SendData(streamName, inPluginId, dataInput)
    if ret < 0:
        print("Failed to send data to stream.")
        exit()

    # Send osd instances protobuf.
    with open("ExternalOsdInstances.json", "r") as f:
        messageJson = json.load(f)
    print(messageJson)
    inPluginId = 1
    osdInstancesList = MxpiOSDType.MxpiOsdInstancesList()
    osdInstancesList = ParseDict(messageJson, osdInstancesList)

    protobufVec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = b'appsrc1'
    protobuf.type = b'MxTools.MxpiOsdInstancesList'
    protobuf.protobuf = osdInstancesList.SerializeToString()
    protobufVec.push_back(protobuf)
    ret = streamManagerApi.SendProtobuf(streamName, inPluginId, protobufVec)
    if ret < 0:
        print("Failed to send protobuf to stream.")
        exit()

    time.sleep(2)

    # destroy streams
    streamManagerApi.DestroyAllStreams()
