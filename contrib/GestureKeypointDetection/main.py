#!/usr/bin/env python
# coding=utf-8

"""

 Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.

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

import sys
import re
import json
import os
import cv2
import random
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

keypointConnectMatrix = [
        [0, 1, 2, 3, 4], 
        [0, 5, 6, 7, 8], 
        [0, 9, 10, 11, 12], 
        [0, 13, 14, 15, 16], 
        [0, 17, 18, 19, 20]
    ]
colorArr = [(0, 215, 255), (255, 115, 55), (5, 255, 55), (25, 15, 255), (225, 15, 55)]
pointColor = (255, 50, 60)

resultCount = 3
decodeResultIndex = 0
handDetectResultIndex = 1
keypointDetectResultIndex = 2
channelCount = 3
maxGray = 255
gestureKeypointCount = 21

YUV_BYTES_NU = 3
YUV_BYTES_DE = 2
EDGE_OFFSET = 3

resizeParameter = 2

def check_range(val, maxVal):
    """check val wether in range
        check val invalid
    """
    if val >= maxVal - EDGE_OFFSET:
        return maxVal - EDGE_OFFSET
    if val > EDGE_OFFSET:
        return val
    return EDGE_OFFSET

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("error input parameters")
        exit()

    filePath = sys.argv[1]

    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    path = b"./detection.pipeline"
    ret = streamManagerApi.CreateMultipleStreamsFromFile(path)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    dataInput = MxDataInput()
    if os.path.exists(filePath) != 1:
        print("The test image does not exist.")
        exit()

    with open(filePath, 'rb') as f:
        dataInput.data = f.read()

    # Inputs data to a specified stream based on streamName.
    streamName = b'detection'
    inPluginId = 0
    ret = streamManagerApi.SendData(streamName, inPluginId, dataInput)
    if ret < 0:
        print("Failed to send data to stream.")
        exit()

    keyVec = StringVector()
    keyVec.push_back(b"mxpi_imagedecoder0")
    keyVec.push_back(b"mxpi_objectpostprocessor0")
    keyVec.push_back(b"mxpi_tensorinfer1")
    infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)

    if len(infer_result) != resultCount:
        print("no hand was detected!")
        streamManagerApi.DestroyAllStreams()
        exit()

    if infer_result.size() == 0:
        print("infer_result is null")
        streamManagerApi.DestroyAllStreams()
        exit()

    if infer_result[keypointDetectResultIndex].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
            infer_result[keypointDetectResultIndex].errorCode, infer_result[keypointDetectResultIndex].data.decode()))
        streamManagerApi.DestroyAllStreams()
        exit()

    tensorList = MxpiDataType.MxpiTensorPackageList()
    tensorList.ParseFromString(infer_result[keypointDetectResultIndex].messageBuf)
    print("len of keypointSet size:", len(tensorList.tensorPackageVec))

    visionList = MxpiDataType.MxpiVisionList()
    visionList.ParseFromString(infer_result[decodeResultIndex].messageBuf)
    visionData = visionList.visionVec[0].visionData.dataStr
    visionInfo = visionList.visionVec[0].visionInfo
    print("len of decode result:", len(visionList.visionVec))

    img_yuv = np.frombuffer(visionData, dtype = np.uint8)
    img_yuv = img_yuv.reshape(visionInfo.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, visionInfo.widthAligned)
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR_NV12)
    img_path = filePath
    img = cv2.imread(img_path)

    pic_height = len(img)
    pic_width = len(img[0])

    mxpiObjectList = MxpiDataType.MxpiObjectList()
    mxpiObjectList.ParseFromString(infer_result[keypointDetectResultIndex].messageBuf)
    print("len of hand size that were detected:", len(mxpiObjectList.objectVec))

    if len(tensorList.tensorPackageVec) != len(mxpiObjectList.objectVec):
        print("keypointSet's size is not equal to the count of hand")
        streamManagerApi.DestroyAllStreams()
        exit()

    # print the infer result
    idsLen = len(tensorList.tensorPackageVec)
    for i in range(idsLen):
        ids = np.frombuffer(tensorList.tensorPackageVec[i].tensorVec[0].dataStr, dtype = np.float32)
        shape = tensorList.tensorPackageVec[i].tensorVec[0].tensorShape
        ids.resize(shape[1] // 2, resizeParameter)

        y0 = int(mxpiObjectList.objectVec[i].y0)
        x0 = int(mxpiObjectList.objectVec[i].x0)
        y1 = int(mxpiObjectList.objectVec[i].y1)
        x1 = int(mxpiObjectList.objectVec[i].x1)
        print(mxpiObjectList.objectVec[i])
        print("y0, x0, y1, x1:", y0, x0, y1, x1)
        recColor = [random.randint(0, maxGray) for _ in range(channelCount)]
        x0Range = check_range(x0, pic_width)
        y0Range = check_range(y0, pic_height)
        x1Range = check_range(x1, pic_width)
        y1Range = check_range(y1, pic_height)
        cv2.rectangle(img, (x0Range, y0Range), (x1Range, y1Range), recColor, 1)

        height = y1 - y0
        width = x1 - x0

        ids = ids * [width, height] + [x0, y0]

        pointArr = []
        for (x, y) in ids.astype(np.int32):
            pointArr.append((x, y))

        if len(pointArr) != gestureKeypointCount:
            print("keypoint size is not equal to 21")
            streamManagerApi.DestroyAllStreams()
            exit()

        colorI = 0
        for link in keypointConnectMatrix:
            linkLen = len(link)
            for index in range(linkLen - 1):
                fromI = link[index]
                toI = link[index + 1]
                ptStart = pointArr[fromI]
                ptEnd = pointArr[toI]
                #draw line
                point_color = colorArr[colorI] # BGR
                thickness = 1 
                lineType = 4
                cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
            colorI = colorI + 1

        for pt in pointArr:
            cv2.circle(img, pt, 2, pointColor, -1)

    buf = filePath
    sub = "/"
    att = [substr.start() for substr in re.finditer(sub, buf)]

    att_len = len(att)
    firstIdx = 0
    if att_len != 0:
        firstIdx = att[len(att) - 1] + 1
    result_name = "result_" + buf[firstIdx:len(buf)]
    print("result name : ", result_name)
    cv2.imwrite("./" + result_name, img)

    streamManagerApi.DestroyAllStreams()
