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
import webcolors
from preprocess import preproc

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector, InProtobufVector, MxProtobufIn

IMAGE_PATH = '../test_img/test.jpg'


def from_colorname_to_bgr(color):
    """
    convert color name to bgr value

    Args:
        color: color name

    Returns: bgr value

    """
    rgb_color = webcolors.name_to_rgb(color)
    resultss = (rgb_color.blue, rgb_color.green, rgb_color.red)
    return resultss


def standard_to_bgr():
    """
    generate bgr list from color name list
    Returns: bgr list

    """
    list_color_name = []
    with open("colorlist.txt", "r") as ff:
        list_color_name = ff.read()
    list_color_name = list_color_name.split(',') 
              
    standard = []
    for i in range(len(list_color_name) - 36):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
        
    return standard


def plot_one_box(origin_img, bbox, color=None, line_thickness=None):
    """
    plot one bounding box on image

    Args:
        origin_img: pending image
        box: infomation of the bounding box
        color: bgr color used to draw bounding box
        line_thickness: line thickness value when drawing the bounding box

    Returns: None

    """
    tl = line_thickness or int(round(0.001 * max(origin_img.shape[0:2])))  # line thickness
    if tl < 1:
        tl = 1
    try:
        c1, c2 = (int(bbox['x0']), int(bbox['y0'])), (int(bbox['x1']), int(bbox['y1']))
        cv2.rectangle(origin_img, c1, c2, color=color, thickness=tl)
        if bbox['text']:
            tf = max(tl - 2, 1)  # font thickness
            s_size = cv2.getTextSize(str('{:.0%}'.format(box['confidence'])), 0, fontScale=float(tl) / 3, thickness=tf)[0]
            t_size = cv2.getTextSize(bbox['text'], 0, fontScale=float(tl) / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
            cv2.rectangle(origin_img, c1, c2, color, -1)  # filled
            cv2.putText(origin_img, '{}: {:.0%}'.format(bbox['text'], bbox['confidence']), (c1[0], c1[1] - 2), 0,
                        float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)
    except KeyError:
        print("error")


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
    color_list = standard_to_bgr()
    for results in objectList.objectVec:
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
            plot_one_box(imgs, box, color=color_list[box.get('class')])
            cv2.imwrite("../test_img/nopre_post.jpg", imgs)
        except KeyError:
            print("error")
    streamManagerApi.DestroyAllStreams()
