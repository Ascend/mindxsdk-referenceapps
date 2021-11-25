#!/usr/bin/env python
# coding=utf-8

"""
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
"""
import argparse
import json
import os
import cv2
import numpy as np
import draw_box
import time
from PIL import Image
from numpy import random


import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-image', type=str, default='test.jpg', help='input image')
    args = parser.parse_args()
    streamManagerApi = StreamManagerApi()
    # 新建一个流管理StreamManager对象并初始化
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()


    with open("./pipeline/rtm3d.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # 构建流的输入对象--检测目标
    dataInput = MxDataInput()
    if os.path.exists(args.input_image) != 1:
        print("The test image does not exist.")

    try:
        with open(args.input_image, 'rb') as f:
            dataInput.data = f.read()
    except FileNotFoundError:
        print("Test image", "test.jpg", "doesn't exist. Exit.")
        exit()
    streamName = b'detection'
    inPluginId = 0
    # 根据流名将检测目标传入流
    uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)

    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()


    keyVec = StringVector()
    keyVec.push_back(b"mxpi_objectpostprocessor0")
    keyVec.push_back(b"mxpi_imageresize0")

    # 从流中取出对应插件的输出数据
    infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)

    if infer_result.size() == 0:
        print("infer_result is null")
        exit()

    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
            infer_result[0].errorCode, infer_result[0].data.decode()))
        exit()

    YUV_BYTES_NU = 3
    YUV_BYTES_DE = 2

    print("infer_result size: ", len(infer_result))

    objectList = MxpiDataType.MxpiObjectList()
    objectList.ParseFromString(infer_result[0].messageBuf)
    print("----------------")
    print(objectList)  # 打印出所有objectinfo


    vision_list0 = MxpiDataType.MxpiVisionList()
    try:
        vision_list0.ParseFromString(infer_result[1].messageBuf)
    except IndexError:
        print("Not find object.")
        exit()
    vision_data0 = vision_list0.visionVec[0].visionData.dataStr
    # Get picture information
    vision_info0 = vision_list0.visionVec[0].visionInfo
    # cv2 func YUV to BGR
    yuv_bytes_nu = 3
    yuv_bytes_de = 2
    img_yuv = np.frombuffer(vision_data0, dtype=np.uint8)
    # reshape
    img_yuv = img_yuv.reshape(vision_info0.heightAligned * yuv_bytes_nu // yuv_bytes_de, vision_info0.widthAligned)
    # Color gamut conversion
    img0 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR_NV12)

    # 类别
    clses = []
    # 置信度
    m_scores = []
    # 框信息
    v_projs_regress = []
    temp = 0
    for results in objectList.objectVec:
        if((temp % 4) == 0):
            clses.append(results.classVec[0].classId)
            m_scores.append(results.classVec[0].confidence)
        v_projs_regress.append(results.x0)
        v_projs_regress.append(results.y0)
        v_projs_regress.append(results.x1)
        v_projs_regress.append(results.y1)
        temp = temp + 1


    clses = np.array(clses)
    m_scores = np.array(m_scores)
    v_projs_regress = np.array(v_projs_regress).reshape(-1, 8, 2)


    if clses[0] is not None:
        draw_box.cv_draw_bbox3d_rtm3d(img0,
                             clses,
                             m_scores,
                             v_projs_regress,
                             label_map=['Car', 'Pedestrian', 'Cyclist']
                             )
        cv2.imwrite('./heatmap.jpg', img0)

    # destroy streams
    streamManagerApi.DestroyAllStreams()
