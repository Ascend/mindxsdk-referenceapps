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


import json
import os
import cv2
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
colorArr = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]]

if __name__ == '__main__':
    streamManagerApi = StreamManagerApi()
    # init stream manager
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    path=b"./test.pipeline"
    ret = streamManagerApi.CreateMultipleStreamsFromFile(path)
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
    # uniqueId = streamManagerApi.SendDataWithUniqueId(streamName, inPluginId, dataInput)
    ret = streamManagerApi.SendData(streamName, inPluginId, dataInput)
    if ret < 0:
        print("Failed to send data to stream.")
        exit()

    # Obtain the inference result by specifying streamName and uniqueId.
    '''
    infer_result = streamManagerApi.GetResultWithUniqueId(streamName, uniqueId, 3000)
    if infer_result.errorCode != 0:
        print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
            infer_result.errorCode, infer_result.data.decode()))
        exit()

    # print the infer result
    print("print the infer result")
    print(infer_result.data.decode())
    '''

    keyVec = StringVector()
    keyVec.push_back(b"mxpi_tensorinfer1")
    keyVec.push_back(b"mxpi_imagedecoder0")
    keyVec.push_back(b"mxpi_objectpostprocessor0")
    infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)

    if infer_result.size() == 0:
        print("infer_result is null")
        exit()

    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
            infer_result[0].errorCode, infer_result[0].data.decode()))
        exit()

    tensorList = MxpiDataType.MxpiTensorPackageList()
    tensorList.ParseFromString(infer_result[0].messageBuf)
    print("len1:", len(tensorList.tensorPackageVec))

    visionList = MxpiDataType.MxpiVisionList()
    visionList.ParseFromString(infer_result[1].messageBuf)
    visionData = visionList.visionVec[0].visionData.dataStr
    visionInfo = visionList.visionVec[0].visionInfo
    print("len2:", len(visionList.visionVec))
    YUV_BYTES_NU = 3
    YUV_BYTES_DE = 2
    img_yuv = np.frombuffer(visionData, dtype = np.uint8)
    img_yuv = img_yuv.reshape(visionInfo.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, visionInfo.widthAligned)
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR_NV12)
    img_path = "test.jpg"
    img = cv2.imread(img_path)

    mxpiObjectList = MxpiDataType.MxpiObjectList()
    mxpiObjectList.ParseFromString(infer_result[2].messageBuf)
    print("len3:", len(mxpiObjectList.objectVec))

    # print the infer result
    idsLen = len(tensorList.tensorPackageVec)
    # idsArr = []
    for i in range(idsLen):
        ids = np.frombuffer(tensorList.tensorPackageVec[i].tensorVec[0].dataStr, dtype = np.float32)
        shape = tensorList.tensorPackageVec[i].tensorVec[0].tensorShape
        ids.resize(shape[1] // 2, 2)

        y0 = mxpiObjectList.objectVec[i].y0
        x0 = mxpiObjectList.objectVec[i].x0
        y1 = mxpiObjectList.objectVec[i].y1
        x1 = mxpiObjectList.objectVec[i].x1
        print(mxpiObjectList.objectVec[i])
        # print("y0, x0, y1, x1:", y0, x0, y1, x1)
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 1)

        height = y1 - y0
        width = x1 - x0

        # expand_width = 1.4 * width
        # x0 = x0 - 0.2 * width
        ids = ids * [width, height] + [x0, y0]

        pointArr = []
        for (x, y) in ids.astype(np.int32):
            cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
            pointArr.append((x, y))

        for link in keypointConnectMatrix:
            linkLen = len(link)
            for index in range(linkLen - 1):
                fromI = link[index]
                toI = link[index + 1]
                ptStart = pointArr[fromI]
                ptEnd = pointArr[toI]
                #draw line
                point_color = (0, 255, 0) # BGR
                thickness = 1 
                lineType = 4
                cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)


    cv2.imwrite("./my_result.jpg", img)

    streamManagerApi.DestroyAllStreams()
    
    # results = json.loads(infer_result.data.decode())
    # bboxes = []

    # img_path = "test.jpg"
    # img = cv2.imread(img_path)

    # for bbox in results['MxpiObject']:
    #     bboxes = {'x0': int(bbox['x0']),
    #               'x1': int(bbox['x1']),
    #               'y0': int(bbox['y0']),
    #               'y1': int(bbox['y1']),
    #               'confidence': round(bbox['classVec'][0]['confidence'], 4),
    #               'text': bbox['classVec'][0]['className']}
    #     text = "{}{}".format(str(bboxes['confidence']), " ")
    #     for item in bboxes['text']:
    #         text += item
    #     cv2.putText(img, text, (bboxes['x0'], bboxes['y0'] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 1)
    #     cv2.rectangle(img, (bboxes['x0'], bboxes['y0']), (bboxes['x1'], bboxes['y1']), (255, 0, 0), 2)
    # cv2.imwrite("./result.jpg", img)
#               ( 0)100, 50
# ( 1)50, 75|( 5)75, 75|( 9)100, 75|(13)125, 75|(17)150, 75
# ( 2)50,100|( 6)75,100|(10)100,100|(14)125,100|(18)150,100
# ( 3)50,125|( 7)75,125|(11)100,125|(15)125,125|(19)150,125
# ( 4)50,150|( 8)75,150|(12)100,150|(16)125,150|(20)150,150

    # keypointArr = [ [100,50],
    #                 [50,75],[50,100],[50,125],[50,150],
    #                 [75,75],[75,100],[75,125],[75,150],
    #                 [100,75],[100,100],[100,125],[100,150],
    #                 [125,75],[125,100],[125,125],[125,150],
    #                 [150,75],[150,100],[150,125],[150,150]
    #                 ]
    # print(keypointConnectMatrix)
    # print(colorArr[1][1])
    # print(keypointArr)

    # img_path = "test.jpg"
    # img = cv2.imread(img_path)

    # for index in range(len(keypointArr)):
    #     print(index)
    #     cv2.circle(img, (keypointArr[index][0], keypointArr[index][1]), 3, (255,0,0), -2)


    # for index in range(len(keypointConnectMatrix)):
    #     for i in range(len(keypointConnectMatrix[index]) - 1):
    #         # cv2.drawLine()
    #         print("one ")
    #     print("---")

    
    # cv2.imwrite("./result.jpg", img)
    # destroy streams
    
