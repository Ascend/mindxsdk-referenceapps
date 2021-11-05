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
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

if __name__ == '__main__':
    emotions = ["anger", "disgust", "fear", "happy", "sad", "surprised", "normal"]
    streamManagerApi = StreamManagerApi()
    # init stream manager
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    pipeline_path = b"./pipeline/facial_expression_recognition.pipeline"
    ret = streamManagerApi.CreateMultipleStreamsFromFile(pipeline_path)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()


    img_path = "image/test4.jpg"
    streamName = b"detection"
    inPluginId = 0
    dataInput = MxDataInput()
    with open(img_path, 'rb') as f:
        dataInput.data = f.read()
    ret = streamManagerApi.SendData(streamName, inPluginId, dataInput)
    if ret < 0:
        print("Failed to send data to stream")
        exit()
    
    keyVec = StringVector()
    keyVec.push_back(b"mxpi_imagedecoder0")
    keyVec.push_back(b"mxpi_distributor0_0")
    keyVec.push_back(b"mxpi_imagecrop0")
    keyVec.push_back(b"mxpi_tensorinfer1")
    infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)

    if infer_result.size() == 0:
        print("infer_result is null")
        exit()

    tensorList3 = MxpiDataType.MxpiTensorPackageList()
    tensorList3.ParseFromString(infer_result[3].messageBuf)

    # print the infer result
    res1 = np.frombuffer(tensorList3.tensorPackageVec[0].tensorVec[0].dataStr, dtype = np.float32)
    maxindex = np.argmax(res1)
    

    visionList0 = MxpiDataType.MxpiVisionList()
    visionList0.ParseFromString(infer_result[2].messageBuf)
    visionData0 = visionList0.visionVec[0].visionData.dataStr
    visionInfo0 = visionList0.visionVec[0].visionInfo


    visionList = MxpiDataType.MxpiVisionList()
    visionList.ParseFromString(infer_result[0].messageBuf)
    visionData = visionList.visionVec[0].visionData.dataStr
    visionInfo = visionList.visionVec[0].visionInfo

    YUV_BYTES_NU = 3
    YUV_BYTES_DE = 2
    img_yuv0 = np.frombuffer(visionData0, dtype = np.uint8)
    img_yuv0 = img_yuv0.reshape(visionInfo0.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, visionInfo0.widthAligned)
    img0 = cv2.cvtColor(img_yuv0, cv2.COLOR_YUV2BGR_NV12)
    cv2.imwrite("./crop_result.jpg", img0)



    img_yuv = np.frombuffer(visionData, dtype = np.uint8)
    img_yuv = img_yuv.reshape(visionInfo.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, visionInfo.widthAligned)
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR_NV12)


    mxpiObjectList = MxpiDataType.MxpiObjectList()
    mxpiObjectList.ParseFromString(infer_result[1].messageBuf)
    y0 = mxpiObjectList.objectVec[0].y0
    x0 = mxpiObjectList.objectVec[0].x0
    y1 = mxpiObjectList.objectVec[0].y1
    x1 = mxpiObjectList.objectVec[0].x1
    height = y1 - y0
    width = x1 - x0
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)
    cv2.putText(img, emotions[maxindex], (int(x0), int(y0)-1), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
    cv2.imwrite("./my_result.jpg", img)
    
    # destroy streams
    streamManagerApi.DestroyAllStreams()
