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

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import *
import json
import cv2
import numpy as np
if __name__ == '__main__':
    pipeline_path = "../pipeline/KeywordDetection.pipeline"
    streamName = b'KeywordDetection'
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(pipeline_path, 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
        
    inplugin_id = 0    
    # Construct the input of the stream
    dataInput = MxDataInput()
    img_path = "../data/en_text/1.jpg"
    
    
    with open(img_path, 'rb') as f:
        dataInput.data = f.read()
    inplugin_id = 0
    uniqueId = streamManagerApi.SendData(streamName, inplugin_id, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()
    
    inplugin_key = 1
    #key
    key_file = open("./bert_key.txt", 'r')
    key_dict = []
    
    for key in key_file.readlines():
        key_dict.append(key.strip())  
    
    mxpiTextsInfoList_key = MxpiDataType.MxpiTextsInfoList()
    textsInfoVec_key = mxpiTextsInfoList_key.textsInfoVec.add()
    
    for key in key_dict:
        textsInfoVec_key.text.append(key)
        
    key1 = b'appsrc1'
    protobufVec = InProtobufVector()
    protobuf_key = MxProtobufIn()
    protobuf_key.key = key1
    protobuf_key.type = b'MxTools.MxpiTextsInfoList'
    protobuf_key.protobuf = mxpiTextsInfoList_key.SerializeToString()
    protobufVec.push_back(protobuf_key)
    

    uniqueId = streamManagerApi.SendProtobuf(streamName, inplugin_key, protobufVec)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    keyVec = StringVector()
    keyVec.push_back(b'appsrc0')
    inferResult = streamManagerApi.GetProtobuf(streamName, inplugin_id, keyVec)
    if inferResult.size() == 0:
        print("inferResult is null")
        exit()
    if inferResult[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" % (
            inferResult[0].errorCode))
        exit()
    streamManagerApi.DestroyAllStreams()
