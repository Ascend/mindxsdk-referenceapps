#!/usr/bin/env python
# coding=utf-8

# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import cv2
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

HEIGHT = 320
WIDTH = 480
DE_NORM = 255

if __name__ == '__main__':
    streamManagerApi = StreamManagerApi()
    # 新建一个流管理StreamManager对象并初始化
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # 构建pipeline
    PIPELINE_PATH = "./t.pipeline"
    if os.path.exists(PIPELINE_PATH) != 1:
        print("Pipeline does not exist !")
        exit()
    with open(PIPELINE_PATH, 'rb') as f:
        pipelineStr = f.read()
        ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
        if ret != 0:
            print("Failed to create Stream, ret=%s" % str(ret))
            exit()

    # 构建流的输入对象--检测目标
    dataInput = MxDataInput()
    IMAGE_PATH = './test.jpg'
    with open(IMAGE_PATH, 'rb') as f:
        dataInput.data = f.read()
        begin_array = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
        h, w = begin_array.shape[:2]
        STREAMNAME = b'detection'
        INPLUGINID = 0
    # 根据流名将检测目标传入流中
    uniqueId = streamManagerApi.SendData(STREAMNAME, INPLUGINID, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()
    keys = [b"mxpi_tensorinfer0"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)
    # 从流中取出对应插件的输出数据
    infer = streamManagerApi.GetResult(STREAMNAME, b'appsink0', keyVec)
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
    pred = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
    pred.resize(HEIGHT+1, WIDTH+1)
    preds = np.zeros((HEIGHT, WIDTH))
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if(pred[i+1][j+1] < 0):
                preds[i][j] = 0
            elif(pred[i+1][j+1] > 1):
                preds[i][j] = DE_NORM
            else:
                preds[i][j] = pred[i+1][j+1] * DE_NORM
    end_array = np.array(preds, dtype=int)
    SAVE_PATH = './result.jpg'
    img_resize = cv2.resize(end_array, (w, h), interpolation = cv2.INTER_NEAREST)
    cv2.imwrite(SAVE_PATH, img_resize)
    # destroy streams
    streamManagerApi.DestroyAllStreams()