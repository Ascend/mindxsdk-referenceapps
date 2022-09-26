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
import webcolors
import numpy as np

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

IMAGE_PATH = "../test_img/test.jpg"


def from_colorname_to_bgr(color):
    """
    convert color name to bgr value

    Args:
        color: color name

    Returns: bgr value

    """
    rgb_color = webcolors.name_to_rgb(color)
    result = (rgb_color.blue, rgb_color.green, rgb_color.red)
    return result


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


def plot_one_box(origin_img, box, color=None, line_thickness=None):
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
        c1, c2 = (int(box['x0']), int(box['y0'])), (int(box['x1']), int(box['y1']))
        cv2.rectangle(origin_img, c1, c2, color=color, thickness=tl)
        if box['text']:
            tf = max(tl - 2, 1)  # font thickness
            s_size = cv2.getTextSize(str('{:.0%}'.format(box['confidence'])), 0, fontScale=float(tl) / 3, 
                                    thickness=tf)[0]
            t_size = cv2.getTextSize(box['text'], 0, fontScale=float(tl) / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
            cv2.rectangle(origin_img, c1, c2, color, -1)  # filled
            cv2.putText(origin_img, '{}: {:.0%}'.format(box['text'], box['confidence']), (c1[0], c1[1] - 2), 0,
                        float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)
    except KeyError:
        print("error")

if __name__ == '__main__':
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    with open("../pipeline/pre_post.pipeline", 'rb') as f:
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
                  'classId': int(results.classVec[0].classId),
                  'text': results.classVec[0].className}
        try:

            TEXT = "{}{}".format(str(bboxes['confidence']), " ")
            for item in bboxes['text']:
                TEXT += item
            color_list = standard_to_bgr()         
            plot_one_box(imgs, bboxes, color=color_list[bboxes.get('classId')]) 
            cv2.imwrite("../test_img/pre_post.jpg", imgs)
        except KeyError:
            print("error")

    streamManagerApi.DestroyAllStreams()
