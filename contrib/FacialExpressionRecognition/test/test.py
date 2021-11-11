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
import cv2
import csv
from PIL import Image
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

if __name__ == '__main__':
    emotions = ["surprised", "fear", "disgust", "happy", "sad", "anger", "normal"]
    streamManagerApi = StreamManagerApi()
    # init stream manager
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    pipeline_path = b"../pipeline/test.pipeline"
    ret = streamManagerApi.CreateMultipleStreamsFromFile(pipeline_path)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    count = 0
    file_name = '../label.txt'
    ima_name = []
    labels = []
    with open(file_name, 'r') as f:
        data = f.readlines()
    for index, _ in enumerate(data):
        line = data[index]
        temp = line.split(' ')
        ima_name.append(temp[0].split('.')[0]+'_aligned.jpg')
        labels.append(int(temp[1]))
    nums = 0
    for index, _ in enumerate(ima_name):
        nums += 1
        streamName = b"detection"
        inPluginId = 0
        dataInput = MxDataInput()
        try:
            with open(img_path, 'rb') as f:
                dataInput.data = f.read()
        except:
            print("No such image")
            exit()
        ret = streamManagerApi.SendData(streamName, inPluginId, dataInput)
        if ret < 0:
            print("Failed to send data to stream")
            exit()
        keyVec = StringVector()
        keyVec.push_back(b"mxpi_tensorinfer1")
        infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)

        if infer_result.size() == 0:
            print("infer_result is null")
            exit()

        tensorList3 = MxpiDataType.MxpiTensorPackageList()
        tensorList3.ParseFromString(infer_result[0].messageBuf)

        # print the infer result
        res1 = np.frombuffer(tensorList3.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        maxindex = np.argmax(res1)
        if(int(maxindex) + 1 == int(labels[index])):
            count = count + 1
        print(int(maxindex) + 1, "***********************", int(labels[index]))
    print("***********************", count / nums, nums)
    
    # destroy streams
    streamManagerApi.DestroyAllStreams()
