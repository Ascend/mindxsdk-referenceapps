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
import cv2
import numpy as np
import webcolors

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

# Color names used to mark detection boxes of different categories
STANDARD_COLORS = [
    'LawnGreen', 'Chartreuse', 'Aqua', 'Beige', 'Azure', 'BlanchedAlmond', 'Bisque',
    'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'AliceBlue', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

# All the class names of the detection target
OBJECT_LIST = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush']


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


def standard_to_bgr(list_color_name):
    """
    generate bgr list from color name list

    Args:
        list_color_name: color name list

    Returns: bgr list

    """
    standard = []
    for i in range(len(list_color_name) - 36):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard


def plot_one_box(origin_img, coord, label=None, box_score=None, color=None, line_thickness=None):
    """
    plot one bounding box on image

    Args:
        origin_img: pending image
        coord: coordinate of bounding box
        label: class label name of the bounding box
        box_score: confidence score of the bounding box
        color: bgr color used to draw bounding box
        line_thickness: line thickness value when drawing the bounding box

    Returns: None

    """
    tl = line_thickness or int(round(0.001 * max(origin_img.shape[0:2])))  # line thickness
    if tl < 1:
        tl = 1
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(origin_img, c1, c2, color=color, thickness=tl)
    if label:
        tf = max(tl - 2, 1)  # font thickness
        s_size = cv2.getTextSize(str('{:.0%}'.format(box_score)), 0, fontScale=float(tl) / 3, thickness=tf)[0]
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
        cv2.rectangle(origin_img, c1, c2, color, -1)  # filled
        cv2.putText(origin_img, '{}: {:.0%}'.format(label, box_score), (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
                    thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)


if __name__ == '__main__':
    streamManagerApi = StreamManagerApi()
    # init stream manager
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("pipeline/EfficientDet-d6.pipeline", "rb") as f:
        pipeline_str = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    dataInput = MxDataInput()
    fileName = 'img.jpg'
    if os.path.exists(fileName) != 1:
        print("The test image does not exist. Exit.")
        exit()
    with open(fileName, 'rb') as f:
        dataInput.data = f.read()
    streamName = b'detection'
    inPluginId = 0
    uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()
    keys = [b"mxpi_objectpostprocessor0"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)
    infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)
    if infer_result.size() == 0:
        print("No object detected")
        img = cv2.imread(fileName)
        cv2.imwrite(fileName.split('.')[0] + "_detect_result.jpg", img)
        exit()
    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
            infer_result[0].errorCode, infer_result[0].data.decode()))
        exit()
    # process data output from mxpi_objectpostprocessor plugin
    objectList = MxpiDataType.MxpiObjectList()
    objectList.ParseFromString(infer_result[0].messageBuf)
    bboxes = []
    for obj in objectList.objectVec:
        box = {'x0': int(obj.x0),
               'x1': int(obj.x1),
               'y0': int(obj.y0),
               'y1': int(obj.y1),
               'class': int(obj.classVec[0].classId),
               'confidence': round(obj.classVec[0].confidence, 4)}
        bboxes.append(box)
    color_list = standard_to_bgr(STANDARD_COLORS)
    img = cv2.imread(fileName)
    # draw each bounding box on the original image
    for box in bboxes:
        class_id = box['class']
        class_name = OBJECT_LIST[class_id]
        score = box['confidence']
        plot_one_box(img, [box['x0'], box['y0'], box['x1'], box['y1']], label=class_name, box_score=score,
                     color=color_list[class_id])
    cv2.imwrite(fileName.split('.')[0] + "_detect_result.jpg", img)

    # destroy streams
    streamManagerApi.DestroyAllStreams()

