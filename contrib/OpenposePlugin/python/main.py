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
import sys
import os
import numpy as np
import cv2
from enum import Enum
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
sys.path.append("../proto")
import mxpiOpenposeProto_pb2 as mxpiOpenposeProto


class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18


COCO_PAIRS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
             (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)]  # = 19

COCO_PAIRS_RENDER = COCO_PAIRS[:-2]

COCO_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


# Visualize skeleton information in a single input image
def visualizeSingleResult(file_name, stream_manager_api, stream_name, in_plugin_id, data_input):
    if os.path.exists(file_name) != 1:
        print("The test image does not exist.")
    with open(file_name, 'rb') as f:
        data_input.data = f.read()
    unique_id = stream_manager_api.SendData(stream_name, in_plugin_id, data_input)
    if unique_id < 0:
        print("Failed to send data to stream.")
        exit()
    key_vec = StringVector()
    key_vec.push_back(b"mxpi_openposeplugin0")
    infer_result = stream_manager_api.GetProtobuf(stream_name, in_plugin_id, key_vec)
    if infer_result.size() == 0:
        print("infer_result is null")
        exit()
    if infer_result[0].errorCode != 0:
        print("infer_result error. errorCode=%d" % (infer_result[0].errorCode))
        exit()
    # print the infer result
    print("GetProtobuf errorCode=%d" % (infer_result[0].errorCode))
    print("KEY: {}".format(str(infer_result[0].messageName)))

    result_personlist = mxpiOpenposeProto.MxpiPersonList()
    result_personlist.ParseFromString(infer_result[0].messageBuf)
    person_list = result_personlist.personInfoVec
    image = cv2.imread(file_name)
    image_show = drawPoseBbox(image, person_list)[0]
    cv2.imwrite(file_name.split('.')[0] + "_single_result.jpg", image_show)
    print("Save Result!")


# draw person keypoints and skeletons
def drawPoseBbox(npimg, person_list, imgcopy=False):
    if imgcopy:
        npimg = np.copy(npimg)
    image_h, image_w = npimg.shape[:2]
    joints, xcenter = [], []
    for person in person_list:
        skeletons = person.skeletonInfoVec
        xs, ys, centers = [], [], {}
        seen_idx = []
        # draw keypoints
        for skele in skeletons:
            part_idx1 = skele.cocoSkeletonIndex1
            part_idx2 = skele.cocoSkeletonIndex2
            if part_idx1 not in seen_idx:
                seen_idx.append(part_idx1)
                center = (int(skele.x0 * image_w + 0.5), int(skele.y0 * image_h + 0.5))
                centers[part_idx1] = center
                xs.append(center[0])
                ys.append(center[1])
                cv2.circle(npimg, center, 3, COCO_COLORS[part_idx1], thickness=3, lineType=8, shift=0)

            if part_idx2 not in seen_idx:
                seen_idx.append(part_idx2)
                center = (int(skele.x1 * image_w + 0.5), int(skele.y1 * image_h + 0.5))
                centers[part_idx2] = center
                xs.append(center[0])
                ys.append(center[1])
                cv2.circle(npimg, center, 3, COCO_COLORS[part_idx2], thickness=3, lineType=8, shift=0)

        # draw skeletons
        for pair_order, pair in enumerate(COCO_PAIRS_RENDER):
            if pair[0] not in seen_idx or pair[1] not in seen_idx:
                continue
            cv2.line(npimg, centers[pair[0]], centers[pair[1]], COCO_COLORS[pair_order], 3, cv2.LINE_AA)

        joints.append(centers)
        if 1 in centers:
            xcenter.append(centers[1][0])

    return npimg, joints


if __name__ == '__main__':
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./pipeline/OpenposePlugin.pipeline", "rb") as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    stream_name = b"classification+detection"
    in_plugin_id = 0
    # Construct the input of the stream
    data_input = MxDataInput()
    visualizeSingleResult("test.jpg", stream_manager_api, stream_name, in_plugin_id, data_input)
    # destroy streams
    stream_manager_api.DestroyAllStreams()
