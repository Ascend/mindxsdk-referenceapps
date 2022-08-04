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

IMAGE_PATH = "../test_img/test.jpg"

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
    dataInput = MxDataInput()
    if os.path.exists(IMAGE_PATH) != 1:
        print("The test image does not exist.")

    with open(IMAGE_PATH, 'rb') as f:
        dataInput.data = f.read()
    imgs = cv2.imread(IMAGE_PATH)
    STREAM_NAME  = b'detection'
    INPLUGIN_ID = 0
    uniqueId = streamManagerApi.SendData(STREAM_NAME, INPLUGIN_ID, dataInput)

    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    keys = [b"mxpi_objectpostprocessor0"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)

    infer_result = streamManagerApi.GetProtobuf(STREAM_NAME, 0, keyVec)

    if infer_result.size() == 0:
        print("infer_result is null")
        exit()

    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" % (
            infer_result[0].errorCode))
        exit()

    objectList = MxpiDataType.MxpiObjectList()
    objectList.ParseFromString(infer_result[0].messageBuf)
    for results in objectList.objectVec:
        if results.classVec[0].classId == 81:
            break
        bboxes = []
        bboxes = {'x0': int(results.x0),
                  'x1': int(results.x1),
                  'y0': int(results.y0),
                  'y1': int(results.y1),
                  'confidence': round(results.classVec[0].confidence, 4),
                  'text': results.classVec[0].className}
        try:

            TEXT = "{}{}".format(str(bboxes['confidence']), " ")
            for item in bboxes['text']:
                TEXT += item
            
            
            cv2.putText(imgs, TEXT, (bboxes['x0'] + 10, bboxes['y0'] + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 1)
            cv2.rectangle(imgs, (bboxes['x0'], bboxes['y0']), (bboxes['x1'], bboxes['y1']), (255, 0, 0), 2)
            cv2.imwrite("../test_img/nopre_post.jpg", imgs)
        except KeyError:
            print("error")
    streamManagerApi.DestroyAllStreams()
