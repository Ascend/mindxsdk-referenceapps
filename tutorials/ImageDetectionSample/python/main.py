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
from StreamManagerApi import StreamManagerApi, MxDataInput

if __name__ == '__main__':
    streamManagerApi = StreamManagerApi()
    # init stream manager
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    pipeline = {
        "detection": {
            "stream_config": {
                "deviceId": "0"
            },
            "mxpi_imagedecoder0": {
                "factory": "mxpi_imagedecoder",
                "next": "mxpi_imageresize0"
            },
            "mxpi_imageresize0": {
                "factory": "mxpi_imageresize",
                "next": "mxpi_modelinfer0"
            },
            "mxpi_modelinfer0": {
                "props": {
                    "modelPath": "models/yolov3_tf_bs1_fp16.om",
                    "postProcessConfigPath": "models/yolov3_tf_bs1_fp16.cfg",
                    "labelPath": "models/coco.names",
                    "postProcessLibPath": "libMpYOLOv3PostProcessor.so"
                },
                "factory": "mxpi_modelinfer",
                "next": "mxpi_imagecrop0"
            },
            "mxpi_imagecrop0": {
                "factory": "mxpi_imagecrop",
                "next": "mxpi_dataserialize0"
            },
            "mxpi_dataserialize0": {
                "props": {
                    "outputDataKeys": "mxpi_modelinfer0"
                },
                "factory": "mxpi_dataserialize",
                "next": "appsink0"
            },
            "appsrc0": {
                "props": {
                    "blocksize": "409600"
                },
                "factory": "appsrc",
                "next": "mxpi_imagedecoder0"
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

    # Construct the input of the stream
    dataInput = MxDataInput()
    if os.path.exists('test.jpg') != 1:
        print("The test image does not exist.")

    with open("test.jpg", 'rb') as f:
        dataInput.data = f.read()

    # Inputs data to a specified stream based on streamName.
    streamName = b'detection'
    inPluginId = 0
    uniqueId = streamManagerApi.SendDataWithUniqueId(streamName, inPluginId, dataInput)

    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    # Obtain the inference result by specifying streamName and uniqueId.
    infer_result = streamManagerApi.GetResultWithUniqueId(streamName, uniqueId, 3000)
    if infer_result.errorCode != 0:
        print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
            infer_result.errorCode, infer_result.data.decode()))
        exit()

    # print the infer result
    print(infer_result.data.decode())

    results = json.loads(infer_result.data.decode())
    bboxes = []
    for bbox in results['MxpiObject']:
        bboxes = {'x0': int(bbox['x0']),
                  'x1': int(bbox['x1']),
                  'y0': int(bbox['y0']),
                  'y1': int(bbox['y1']),
                  'confidence': round(bbox['classVec'][0]['confidence'], 4),
                  'text': bbox['classVec'][0]['className']}
    img_path = "test.jpg"
    img = cv2.imread(img_path)
    text = "{}{}".format(str(bboxes['confidence']), " ")

    for item in bboxes['text']:
        text += item
    cv2.putText(img, text, (bboxes['x0'] + 10, bboxes['y0'] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 1)
    cv2.rectangle(img, (bboxes['x0'], bboxes['y0']), (bboxes['x1'], bboxes['y1']), (255, 0, 0), 2)

    cv2.imwrite("./result.jpg", img)

    # destroy streams
    streamManagerApi.DestroyAllStreams()
