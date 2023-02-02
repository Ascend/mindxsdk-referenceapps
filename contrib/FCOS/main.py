#!/usr/bin/env python
# coding=utf-8
# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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
import time
import cv2
import numpy as np
import mmcv
import webcolors
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxProtobufIn, InProtobufVector, StringVector


def from_colorname_to_bgr(color):
    rgb_color = webcolors.name_to_rgb(color)
    result_change = (rgb_color.blue, rgb_color.green, rgb_color.red)
    return result_change


def standard_to_bgr():
    list_color_name = []
    with open("colorlist.txt", "r") as ff:
        list_color_name = ff.read()
    list_color_name = list_color_name.split(',')

    standard = []
    for i in range(len(list_color_name) -
                   36):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))

    return standard


def plot_one_box(origin_img, box, color=None, line_thickness=None):
    tl = line_thickness or int(round(
        0.001 * max(origin_img.shape[0:2])))  # line thickness
    if tl < 1:
        tl = 1
    c1, c2 = (int(box['x0']), int(box['y0'])), (int(box['x1']), int(box['y1']))
    cv2.rectangle(origin_img, c1, c2, color=color, thickness=tl)
    if box['text']:
        tf = max(tl - 2, 1)  # font thickness
        s_size = cv2.getTextSize(str('{:.4%}'.format(box['confidence'])),
                                 0,
                                 fontScale=float(tl) / 3,
                                 thickness=tf)[0]
        t_size = cv2.getTextSize(box['text'],
                                 0,
                                 fontScale=float(tl) / 3,
                                 thickness=tf)[0]
        c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
        cv2.rectangle(origin_img, c1, c2, color, -1)  # filled
        cv2.putText(origin_img,
                    '{}: {:.4%}'.format(box['text'], box['confidence']),
                    (c1[0], c1[1] - 2),
                    0,
                    float(tl) / 3, [0, 0, 0],
                    thickness=tf,
                    lineType=cv2.FONT_HERSHEY_SIMPLEX)


def resize_image(image, size):
    old_h = image.shape[0]
    old_w = image.shape[1]
    scale_ratio = min(size[0] / old_w, size[1] / old_h)
    new_w = int(np.floor(old_w * scale_ratio))
    new_h = int(np.floor(old_h * scale_ratio))
    image = mmcv.imresize(image, (new_w, new_h), backend='cv2')
    return image, scale_ratio


def preprocess(imagepath):
    image = mmcv.imread(imagepath, backend='cv2')
    image, scale = resize_image(image, (1333, 800))
    h = image.shape[0]
    w = image.shape[1]
    mean = np.array([102.9801, 115.9465, 122.7717], dtype=np.float32)
    std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    norm_img_data = mmcv.imnormalize(image, mean, std, to_rgb=False)
    padleft = (1333 - w) // 2
    padtop = (800 - h) // 2
    padright = 1333 - w - padleft
    padbottom = 800 - h - padtop
    image_for_infer = mmcv.impad(norm_img_data,
                                 padding=(padleft, padtop, padright,
                                          padbottom),
                                 pad_val=0)
    image_for_infer = image_for_infer.transpose(2, 0, 1)
    return image_for_infer, [scale, padleft, padtop, padright, padbottom]


if __name__ == '__main__':
    streamManagerApi = StreamManagerApi()
    # create a StreamManager and init it
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create pipeline
    with open("./pipeline/FCOSdetection.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    IMAGENAME = '{image path}'
    # create detection object
    if os.path.exists(IMAGENAME) != 1:
        print("The test image does not exist.")
        exit()

    # get the size of picture.
    tensor_data, return_image = preprocess(IMAGENAME)
    tensor = tensor_data[None, :]

    visionList = MxpiDataType.MxpiVisionList()
    visionVec = visionList.visionVec.add()
    visionInfo = visionVec.visionInfo
    # The standard input size of FCOS model is 1333*800
    visionInfo.width = 1333
    visionInfo.height = 800
    visionInfo.widthAligned = 1333
    visionInfo.heightAligned = 800
    visionData = visionVec.visionData
    visionData.dataStr = tensor.tobytes()
    visionData.deviceId = 0
    visionData.memType = 0
    visionData.dataSize = len(tensor)

    STREAMNAME = b'detection'
    INPLUGINID = 0

    # put the detection object to stream
    KEY_VALUE = b"appsrc0"
    protobufVec = InProtobufVector()

    protobuf = MxProtobufIn()
    protobuf.key = KEY_VALUE
    protobuf.type = b"MxTools.MxpiVisionList"
    protobuf.protobuf = visionList.SerializeToString()
    protobufVec.push_back(protobuf)

    uniqueId = streamManagerApi.SendProtobuf(STREAMNAME, INPLUGINID,
                                             protobufVec)

    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    YUV_BYTES_NU = 3
    YUV_BYTES_DE = 2
    keys = [b"mxpi_objectpostprocessor0"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)

    # get the output data from the stream plugin
    infer_result = streamManagerApi.GetProtobuf(STREAMNAME, 0, keyVec)
    if infer_result.size() == 0:
        print("infer_result is null")
        exit()

    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
        exit()

    # get output information from mxpi_objectpostprocessor0
    objectList = MxpiDataType.MxpiObjectList()
    objectList.ParseFromString(infer_result[0].messageBuf)
    print(objectList)

    img = cv2.imread(IMAGENAME)
    result = objectList.objectVec
    myscale, pad_left, pad_top, pad_right, pad_bottom = return_image

    classify = []
    with open('./models/coco.names') as f:
        for line in f.readlines():
            line = line.strip('\n')
            classify.append(str(line))
    color_list = standard_to_bgr()
    for x in result:
        TEXTNAME = "{}{}".format(str(round(x.classVec[0].confidence, 4)), " ")
        TEXTNAME += str(classify[int(x.classVec[0].classId)])
        new_x0 = max(int((x.x0 - pad_left) / myscale), 0)
        new_x1 = max(int((x.x1 - pad_left) / myscale), 0)
        new_y0 = max(int((x.y0 - pad_top) / myscale), 0)
        new_y1 = max(int((x.y1 - pad_top) / myscale), 0)
        objBox = {
            'x0': new_x0,
            'x1': new_x1,
            'y0': new_y0,
            'y1': new_y1,
            'text': str(classify[int(x.classVec[0].classId)]),
            'classId': int(x.classVec[0].classId),
            'confidence': round(x.classVec[0].confidence, 4)
        }
        plot_one_box(img, objBox, color=color_list[objBox.get('classId')])
    cv2.imwrite("./result.jpg", img)
    # destroy streams
    streamManagerApi.DestroyAllStreams()
