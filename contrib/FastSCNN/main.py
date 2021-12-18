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
import cv2
import numpy as np

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

if __name__ == '__main__':
    streamManagerApi = StreamManagerApi()
    # 新建一个流管理StreamManager对象并初始化
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit() 
    # 调色板
    cityscapepallete = [
        128, 64, 128,
        244, 35, 232,
        70, 70, 70,
        102, 102, 156,
        190, 153, 153,
        153, 153, 153,
        250, 170, 30,
        220, 220, 0,
        107, 142, 35,
        152, 251, 152,
        0, 130, 180,
        220, 20, 60,
        255, 0, 0,
        0, 0, 142,
        0, 0, 70,
        0, 60, 100,
        0, 80, 100,
        0, 0, 230,
        119, 11, 32,
    ]

    # 构建pipeline
    pipeline_path = "./text.pipeline"
    if os.path.exists(pipeline_path) != 1:
        print("Pipeline does not exist !")
        exit()
    if pipeline_path.split('.')[-1] != 'pipeline':
        print("Pipeline File Error (end with .pipeline)")
        exit()
    with open(pipeline_path, 'rb') as f:
        pipelineStr = f.read()
        ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
        if ret != 0:
            print("Failed to create Stream, ret=%s" % str(ret))
            exit()

    # 构建流的输入对象
    dataInput = MxDataInput()
    image_path = "./test.jpg"
    if os.path.exists(image_path) != 1:
        print("The test image does not exist ！")
        exit()
    if image_path.split('.')[-1] != 'jpg':
        print("DVPP only support jpg !")
        exit()
    with open(image_path , 'rb') as f:
        orimg = cv2.imread(image_path)
        orig_h = orimg.shape[0]
        orig_w = orimg.shape[1]
        dataInput.data = f.read()
        streamName = b'detection'
        inPluginId = 0
    uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    keys = [b"mxpi_tensorinfer0"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)

    # 从流中取出对应插件的输出数据
    infer = streamManagerApi.GetResult(streamName, b'appsink0', keyVec)
    if(infer.metadataVec.size() == 0):
        print("Get no data from stream !")
        exit()
    print("result.metadata size: ", infer.metadataVec.size())
    infer_result = infer.metadataVec[0]
    if infer_result.errorCode != 0:
        print("GetResult error. errorCode=%d , errMsg=%s" % (infer_result.errorCode, infer_result.errMsg))
        exit()
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result.serializedMetadata)

    pred = np.frombuffer(result.tensorPackageVec[0].tensorVec[2].dataStr
                          , dtype=np.float16)
    WIDTH = 2048
    HEIGHT = 1024
    CLASS = 19
    pred.resize(CLASS, HEIGHT, WIDTH)
    pre = np.argmax(pred , 0) 
    array_pred = np.array(pre, dtype=int)
    img_s = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    img_show = np.array(img_s, dtype=int)
    
    # 涂色
    for i in range(HEIGHT):
        for j in range(WIDTH):
            a = array_pred[i][j]
            img_show[i][j][0] = cityscapepallete[a*3+2]
            img_show[i][j][1] = cityscapepallete[a*3+1]
            img_show[i][j][2] = cityscapepallete[a*3]
    img_resize = cv2.resize(img_show, (orig_w, orig_h), interpolation = cv2.INTER_NEAREST)
    cv2.imwrite("./mask.png" , img_resize)
