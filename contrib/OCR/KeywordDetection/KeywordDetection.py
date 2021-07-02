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
from StreamManagerApi import StreamManagerApi
import json
import cv2
import numpy as np
if __name__ == '__main__':
    pipeline_path = "../pipeline/KeywordDetection.pipeline"
    streamName = b'KeywordDetection'
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(pipeline_path, 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
        
    inplugin_id = 0    
    # Construct the input of the stream
    data_input = MxDataInput()
    img_path = "../data/en_text/1.jpg"
    
    
    with open(img_path, 'rb') as f:
        data_input.data = f.read()
    inplugin_id = 0
    unique_id = stream_manager_api.SendData(streamName, inplugin_id, data_input)
    if unique_id < 0:
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
    protobuf_vec = InProtobufVector()
    protobuf_key = MxProtobufIn()
    protobuf_key.key = key1
    protobuf_key.type = b'MxTools.MxpiTextsInfoList'
    protobuf_key.protobuf = mxpiTextsInfoList_key.SerializeToString()
    protobuf_vec.push_back(protobuf_key)
    

    unique_id = stream_manager_api.SendProtobuf(streamName, inplugin_key, protobuf_vec)
    if unique_id < 0:
        print("Failed to send data to stream.")
        exit()

    key_vec = StringVector()
    key_vec.push_back(b'appsrc0')
    infer_result = stream_manager_api.GetProtobuf(streamName, inplugin_id, key_vec)
    if infer_result.size() == 0:
        print("infer_result is null")
        exit()
    if inferResult[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" % (
            inferResult[0].errorCode))
        exit()
    stream_manager_api.DestroyAllStreams()
