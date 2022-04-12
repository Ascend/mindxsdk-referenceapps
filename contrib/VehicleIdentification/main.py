#!/usr/bin/env python
# coding=utf-8

# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import copy
import math
import cv2
import json
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector


if __name__ == '__main__':
    # Create and initialize a new StreamManager object
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # Read and format pipeline file
    with open("./pipeline/identification.pipeline", 'rb') as f:
        pipeline = f.read().replace(b'\r', b'').replace(b'\n', b'')
    pipelineStr = pipeline

    # Create Stream
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Create Input Object
    dataInput = MxDataInput()
    if os.path.exists('test.jpg') != 1:
        print("The test image does not exist.")

    with open("test.jpg", 'rb') as f:
        dataInput.data = f.read()

    # Stream Info
    streamName = b'identification'
    inPluginId = 0
    # Send Input Data to Stream
    uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)

    # Get the result returned by the plugins
    keys = [b"mxpi_imagedecoder0", b"mxpi_distributor0_0", b"mxpi_tensorinfer1"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)

    outPluginId = 0
    infer_result = streamManagerApi.GetProtobuf(streamName, outPluginId, keyVec)
    
    imgdecoder_result_index = 0
    yolo_result_index = 1
    vehicle_result_index = 2
    
    if infer_result.size() == 0:
        print("infer_result is null")
        image = cv2.imread('test.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_res = copy.deepcopy(image)
        image_res = cv2.cvtColor(image_res, cv2.COLOR_RGB2BGR)
        SRC_PATH = os.path.realpath(__file__).rsplit("/", 1)[0]
        Output_PATH = os.path.join(SRC_PATH, "./test_output.jpg")
        cv2.imwrite(Output_PATH, image_res)
        exit()

    if infer_result[yolo_result_index].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorPlugin=%s" % (
            infer_result[yolo_result_index].errorCode, infer_result[yolo_result_index].messageName))
        exit()

    objectList = MxpiDataType.MxpiObjectList()
    objectList.ParseFromString(infer_result[yolo_result_index].messageBuf)
    print(objectList)
    results = objectList.objectVec

    if infer_result[vehicle_result_index].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorPlugin=%s" % (
            infer_result[vehicle_result_index].errorCode, infer_result[vehicle_result_index].messageName))
        exit()

    tensorList = MxpiDataType.MxpiTensorPackageList()
    tensorList.ParseFromString(infer_result[vehicle_result_index].messageBuf)

    label = open("./models/make_model_names_cls.csv","r")
    class_indict =label.readlines()
    
    YUV_BYTES_NU = 3
    YUV_BYTES_DE = 2
    
    # mxpi_imagedecoder0 图像解码插件输出信息
    visionList = MxpiDataType.MxpiVisionList()
    visionList.ParseFromString(infer_result[imgdecoder_result_index].messageBuf)
    
    vision_data = visionList.visionVec[0].visionData.dataStr
    visionInfo = visionList.visionVec[0].visionInfo

    # 用输出原件信息初始化OpenCV图像信息矩阵
    img_yuv = np.frombuffer(vision_data, np.uint8)

    img_bgr = img_yuv.reshape(visionInfo.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, visionInfo.widthAligned)
    img = cv2.cvtColor(img_bgr, getattr(cv2, "COLOR_YUV2BGR_NV12"))
    
    index = 0
    bboxes = []
    for i, _ in enumerate(tensorList.tensorPackageVec):
        res1 = np.frombuffer(tensorList.tensorPackageVec[i].tensorVec[0].dataStr, dtype = np.float32)
        print(res1.shape)
        maxindex = res1.transpose().argmax()
        print(maxindex)
        maxvalue = res1.transpose().max()
        print(maxvalue)
        print_res = "class: {}   prob: {:.3}".format(class_indict[maxindex+1], maxvalue)
        print(print_res)
        bboxes = {'x0': int(results[index].x0),
                  'x1': int(results[index].x1),
                  'y0': int(results[index].y0),
                  'y1': int(results[index].y1),
                  'confidence': round(maxvalue, 4),
                  'text': class_indict[maxindex]}
        text = "{}:{}".format(bboxes['text'], str(bboxes['confidence']))
        if bboxes['confidence'] > 0.2:
            cv2.putText(img, text, (bboxes['x0'] + 30, bboxes['y0'] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.rectangle(img, (bboxes['x0'], bboxes['y0']), (bboxes['x1'], bboxes['y1']), (255, 0, 0), 3)
        index += 1
    
    cv2.imwrite("./result.jpg", img) 
    # destroy streams
    streamManagerApi.DestroyAllStreams()
