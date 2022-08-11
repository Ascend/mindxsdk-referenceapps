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
from preprocess import preproc


import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector, InProtobufVector, MxProtobufIn

IMAGE_PATH = '../test_img/test.jpg'


if __name__ == '__main__':
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()


    with open("../pipeline/nopre_post.pipeline", 'rb') as f:
        pipelineStr = f.read()

    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    imgs = cv2.imread(IMAGE_PATH)
    if os.path.exists(IMAGE_PATH) != 1:
        print("The test image does not exist. Exit.")
        exit()
    img = cv2.imread(IMAGE_PATH)
    
    pred_img = preproc(img)
    
    visionList = MxpiDataType.MxpiVisionList()
    visionVec = visionList.visionVec.add()
    
    visionInfo = visionVec.visionInfo
    visionInfo.width = img.shape[1]
    visionInfo.height = img.shape[0]
    visionInfo.widthAligned = 512
    visionInfo.heightAligned = 512

    visionData = visionVec.visionData
    visionData.dataStr = pred_img.tobytes()
    visionData.deviceId = 0
    visionData.memType = 0
    visionData.dataSize = len(pred_img)
    KEY0 = b"appsrc0"

    protobufVec = InProtobufVector()

    
    protobuf = MxProtobufIn()
    protobuf.key = KEY0
    protobuf.type = b"MxTools.MxpiVisionList"
    protobuf.protobuf = visionList.SerializeToString()
    protobufVec.push_back(protobuf)


    STREAM_NAME = b'detection'
    INPLUGIN_ID = 0
    uniqueId = streamManagerApi.SendProtobuf(STREAM_NAME, INPLUGIN_ID, protobufVec)

    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    keyVec = StringVector()
    keyVec.push_back(b"mxpi_tensorinfer0")
    keyVec.push_back(b"mxpi_objectpostprocessor0")
    
    inferResult = streamManagerApi.GetResult(STREAM_NAME, b'appsink0', keyVec)
    if inferResult.metadataVec.size() == 0:
        print("GetResult failed")
        exit()


    tensorInfer = inferResult.metadataVec[0]


    if tensorInfer.errorCode != 0:
        print("GetResult error. errorCode=%d, errMsg=%s" % (tensorInfer.errorCode, tensorInfer.errMsg))
        exit()
    tensorResult = MxpiDataType.MxpiTensorPackageList()
    tensorResult.ParseFromString(tensorInfer.serializedMetadata)
    result = []
    for idx in range(len(tensorResult.tensorPackageVec[0].tensorVec)):
        result.append(np.frombuffer(tensorResult.tensorPackageVec[0].tensorVec[idx].dataStr, dtype=np.float32))
    
    tensorInfer = inferResult.metadataVec[1]
    if tensorInfer.errorCode != 0:
        print("GetResult error. errorCode=%d, errMsg=%s" % (tensorInfer.errorCode, tensorInfer.errMsg))
        exit()
    objectList = MxpiDataType.MxpiObjectList()
    objectList.ParseFromString(tensorInfer.serializedMetadata)

    for results in objectList.objectVec:
        print(results)
        if results.classVec[0].classId == 81:
            break
        box = []
        box = {'x0': int(results.x0),
               'x1': int(results.x1),
                'y0': int(results.y0),
                'y1': int(results.y1),
                  'confidence': round(results.classVec[0].confidence, 4),
                  'class': results.classVec[0].classId,
                  'text': results.classVec[0].className}
        try:
            TEXT = "{}{}".format(str(box['confidence']), " ")
            for item in box['text']:
                TEXT += item
            cv2.putText(imgs, TEXT, (box['x0'] + 10, box['y0'] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 1)
            cv2.rectangle(imgs, (box['x0'], box['y0']), (box['x1'], box['y1']), (255, 0, 0), 2)
            cv2.imwrite("../test_img/pre_post.jpg", imgs)
        except KeyError:
            print("error")
    streamManagerApi.DestroyAllStreams()
