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
    count = 0
    inters = np.zeros(19)
    labels = np.zeros(19)
    preds = np.zeros(19)
    unions = np.zeros(19)
    correct = np.zeros(500)
    labeled = np.zeros(500)
    sum_correct = 0
    sum_labeled = 0

    # Id-trainId 索引
    classMap = {
        0: 7,
        1: 8,
        2: 11,
        3: 12,
        4: 13,
        5: 17,
        6: 19,
        7: 20,
        8: 21,
        9: 22,
        10: 23,
        11: 24,
        12: 25,
        13: 26,
        14: 27,
        15: 28,
        16: 31,
        17: 32,
        18: 33
    }
    
    # id
    index_label = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

    # 构建pipeline
    with open("./text.pipeline", 'rb') as f:
        pipelineStr = f.read()
        ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
        if ret != 0:
            print("Failed to create Stream, ret=%s" % str(ret))
            exit()

    # 构建流的输入对象
    dataInput = MxDataInput()
    for filename in os.listdir("./cityscapes/leftImg8bit/val/frankfurt"):
        with open("./cityscapes/leftImg8bit/val/frankfurt/" + filename, 'rb') as f:
            imgpath = "./cityscapes/gtFine/val/frankfurt/Label/" + filename.split('_')[0] + '_' + filename.split('_')[1] \
            + '_' + filename.split('_')[2] + "_gtFine_labelIds.png"
            array_label = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
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
        print("result.metadata size: ", infer.metadataVec.size())
        infer_result = infer.metadataVec[0]
        if infer_result.errorCode != 0:
            print("GetResult error. errorCode=%d , errMsg=%s" % (infer_result.errorCode, infer_result.errMsg))
            exit()
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result.serializedMetadata)

        pred3 = np.frombuffer(result.tensorPackageVec[0].tensorVec[2].dataStr
                          , dtype=np.float16)
        HEIGHT = 1024
        WIDTH = 2048
        CLASS = 19
        pred.resize(CLASS, HEIGHT, WIDTH)
        pre = np.argmax(pred, 0)
        array_pred = np.array(pre, dtype=int)
    
    # 评估结果
        print("Segmentation Evaluation [", count + 1, "] Starts:")
        sum_iou = 0.0

        for i in range(HEIGHT):
            for j in range(WIDTH):
                if (array_label[i][j] in index_label):
                    a = array_pred[i][j]
                    temp = array_label[i][j]
                    for k in range(CLASS):
                        if(classMap[k] == temp):
                            b = k
                            break
                    labels[b] += 1
                    preds[a] += 1
                    labeled[count] += 1
                    if a == b:
                        inters[b] += 1
                        correct[count] += 1          
              
        for i in range(CLASS):
            unions[i] = preds[i] + labels[i] - inters[i]
            if unions[i] != 0:
                print("Class(", i + 1, ") IoU : ", inters[i] * 1.0000 / unions[i])
                sum_iou += inters[i] / unions[i]
            sum_labeled += labeled[count]
            sum_correct += correct[count] 
        print("Model PA: ", sum_correct * 1.0000 / sum_labeled * 100, "%")
        print("Model MIoU: ", sum_iou / CLASS)
        count += 1
    streamManagerApi.DestroyAllStreams()
