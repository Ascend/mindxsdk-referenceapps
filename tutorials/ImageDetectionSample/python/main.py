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

    # 构建pipeline
    pipeline = {
        "detection": {
            "stream_config": {
                "deviceId": "0"
            },
            "appsrc0": {
                "props": {
                    "blocksize": "409600"
                },
                "factory": "appsrc",
                "next": "mxpi_imagedecoder0"
            },
            "mxpi_imagedecoder0": {
                "props": {
                    "deviceId": "0"
                },
                "factory": "mxpi_imagedecoder",
                "next": "mxpi_imageresize0"
            },
            "mxpi_imageresize0": {
                "props": {
                    "dataSource": "mxpi_imagedecoder0",
                    "resizeHeight": "416",
                    "resizeWidth": "416"
                },
                "factory": "mxpi_imageresize",
                "next": "mxpi_tensorinfer0"
            },
            "mxpi_tensorinfer0": {
                "props": {
                    "dataSource": "mxpi_imageresize0",
                    "modelPath": "models/yolov3_tf_bs1_fp16.om"
                },
                "factory": "mxpi_tensorinfer",
                "next": "mxpi_objectpostprocessor0"
            },
            "mxpi_objectpostprocessor0": {
                "props": {
                    "dataSource": "mxpi_tensorinfer0",
                    "postProcessConfigPath": "models/yolov3_tf_bs1_fp16.cfg",
                    "labelPath": "models/coco.names",
                    "postProcessLibPath": "${SDK安装路径}/lib/modelpostprocessors/libyolov3postprocess.so"
                },
                "factory": "mxpi_objectpostprocessor",
                "next": "appsink0"
            },
            "appsink0": {
                "props": {
                    "blocksize": "4096000"
                },
                "factory": "appsink"
            }
        }
    }

    pipelineStr = json.dumps(pipeline).encode()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # 构建流的输入对象--检测目标
    dataInput = MxDataInput()
    if os.path.exists('test.jpg') != 1:
        print("The test image does not exist.")

    with open("test.jpg", 'rb') as f:
        dataInput.data = f.read()

    streamName = b'detection'
    inPluginId = 0
    # 根据流名将检测目标传入流中
    uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)

    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    keys = [b"mxpi_objectpostprocessor0", b"mxpi_imagedecoder0"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)

    # 从流中取出对应插件的输出数据
    infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)

    if infer_result.size() == 0:
        print("infer_result is null")
        exit()

    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" % (
            infer_result[0].errorCode))
        exit()

    YUV_BYTES_NU = 3
    YUV_BYTES_DE = 2
    # mxpi_objectpostprocessor0模型后处理插件输出信息
    objectList = MxpiDataType.MxpiObjectList()
    objectList.ParseFromString(infer_result[0].messageBuf)
    print(objectList)

    # mxpi_imagedecoder0 图像解码插件输出信息
    visionList = MxpiDataType.MxpiVisionList()
    visionList.ParseFromString(infer_result[1].messageBuf)

    vision_data = visionList.visionVec[0].visionData.dataStr
    visionInfo = visionList.visionVec[0].visionInfo

    # 用输出原件信息初始化OpenCV图像信息矩阵
    img_yuv = np.frombuffer(vision_data, np.uint8)

    img_bgr = img_yuv.reshape(visionInfo.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, visionInfo.widthAligned)
    img = cv2.cvtColor(img_bgr, getattr(cv2, "COLOR_YUV2BGR_NV12"))

    # 打印出推理结果
    results = objectList.objectVec[0]

    bboxes = []
    bboxes = {'x0': int(results.x0),
              'x1': int(results.x1),
              'y0': int(results.y0),
              'y1': int(results.y1),
              'confidence': round(results.classVec[0].confidence, 4),
              'text': results.classVec[0].className}

    text = "{}{}".format(str(bboxes['confidence']), " ")

    for item in bboxes['text']:
        text += item
    cv2.putText(img, text, (bboxes['x0'] + 10, bboxes['y0'] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 1)
    cv2.rectangle(img, (bboxes['x0'], bboxes['y0']), (bboxes['x1'], bboxes['y1']), (255, 0, 0), 2)

    cv2.imwrite("./result.jpg", img)

    # destroy streams
    streamManagerApi.DestroyAllStreams()
