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
import cv2
import copy
import math
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
sys.path.append("./plugins/proto")
import mxpiHeadPoseProto_pb2 as mxpiHeadPoseProto


def whenet_draw(yaw, pitch, roll, tdx=None, tdy=None, size=200):
    """
    Plot lines based on yaw pitch roll values

    Args:
        yaw, pitch, roll: values of angles
        tdx, tdy: center of detected head area

    Returns:
        graph: locations of three lines
    """
    # taken from hopenet
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    tdx = tdx
    tdy = tdy

    # X-Axis pointing to right. drawn in red
    x1 = size * (math.cos(yaw) * math.cos(roll)) + tdx
    y1 = size * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(yaw)) + tdy

    # Y-Axis | drawn in green
    x2 = size * (-math.cos(yaw) * math.sin(roll)) + tdx
    y2 = size * (math.cos(pitch) * math.cos(roll) - math.sin(pitch) * math.sin(yaw) * math.sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (math.sin(yaw)) + tdx
    y3 = size * (-math.cos(yaw) * math.sin(pitch)) + tdy
    
    return {
        "yaw_x": x1,
        "yaw_y": y1, 
        "pitch_x": x2, 
        "pitch_y": y2, 
        "roll_x": x3, 
        "roll_y": y3
    }


if __name__ == '__main__':
    # Create and initialize a new StreamManager object
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # Read and format pipeline file
    with open("./pipeline/recognition.pipeline", 'rb') as f:
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
    streamName = b'recognition'
    inPluginId = 0
    # Send Input Data to Stream
    uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)

    # Get the result returned by the plugins
    keys = [b"mxpi_objectpostprocessor0", b"mxpi_headposeplugin0"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)

    outPluginId = 0
    infer_result = streamManagerApi.GetProtobuf(streamName, outPluginId, keyVec)

    yolo_result_index = 0
    whenet_result_index = 1
    if infer_result.size() == 0:
        print("infer_result is null")
        exit()

    if infer_result[yolo_result_index].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorPlugin=%s" % (
            infer_result[yolo_result_index].errorCode, infer_result[yolo_result_index].messageName))
        exit()

    objectList = MxpiDataType.MxpiObjectList()
    objectList.ParseFromString(infer_result[yolo_result_index].messageBuf)
    print(objectList)
    results = objectList.objectVec[0]

    if infer_result[whenet_result_index].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorPlugin=%s" % (
            infer_result[whenet_result_index].errorCode, infer_result[whenet_result_index].messageName))
        exit()

    result_protolist = mxpiHeadPoseProto.MxpiHeadPoseList()
    result_protolist.ParseFromString(infer_result[whenet_result_index].messageBuf)
    print("YAW:")
    print("result: {}".format(
        result_protolist.headposeInfoVec[0].yaw))
    print("PITCH:")
    print("result: {}".format(
        result_protolist.headposeInfoVec[0].pitch))
    print("ROLL:")
    print("result: {}".format(
        result_protolist.headposeInfoVec[0].roll))

    yaw_predicted = result_protolist.headposeInfoVec[0].yaw
    pitch_predicted = result_protolist.headposeInfoVec[0].pitch
    roll_predicted = result_protolist.headposeInfoVec[0].roll

    image = cv2.imread('test.jpg')
    image_height, image_width = image.shape[0], image.shape[1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    box_width = (results.x0 + results.x1)/2
    box_height = (results.y0 + results.y1)/2
    detection_item = whenet_draw(yaw_predicted, pitch_predicted, roll_predicted,
                                 tdx=box_width, tdy=box_height, size=100)

    save_img = True
    if save_img:
        image_res = copy.deepcopy(image)
    # plot head detection box from yolo predictions
    line_thickness = 4
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)

    cv2.rectangle(image_res, (int(results.x0 - ((results.x1 - results.x0) * 0.25)),
                              int(results.y0 - ((results.y1 - results.y0) * 0.35))),
                             (int(results.x1 + ((results.x1 - results.x0) * 0.25)),
                              int(results.y1 + ((results.y1 - results.y0) * 0.1))),
                             (127, 125, 125), 2)
    # plot head pose detection lines from whenet predictions
    cv2.line(image_res, (int(box_width), int(box_height)),
             (int(detection_item["yaw_x"]), int(detection_item["yaw_y"])), red, line_thickness)
    cv2.line(image_res, (int(box_width), int(box_height)),
             (int(detection_item["pitch_x"]), int(detection_item["pitch_y"])), green, line_thickness)
    cv2.line(image_res, (int(box_width), int(box_height)),
             (int(detection_item["roll_x"]), int(detection_item["roll_y"])), blue, line_thickness)

    if save_img:
        image_res = cv2.cvtColor(image_res, cv2.COLOR_RGB2BGR)
        SRC_PATH = os.path.realpath(__file__).rsplit("/", 1)[0]
        Output_PATH = os.path.join(SRC_PATH, "./test_output.jpg")
        cv2.imwrite(Output_PATH, image_res)
    # destroy streams
    streamManagerApi.DestroyAllStreams()
