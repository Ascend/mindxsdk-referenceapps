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

import os
import cv2
import numpy as np
import sys
sys.path.append('.')
sys.path.append('../')

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

from faceswap_post import *

FACE1_PATH =  sys.argv[1]
FACE2_PATH =  sys.argv[2]

STREAM_NAME = b'faceswap'
IN_PLUGIN_ID = 0

POINTS_NUMS = 106  # there are 106 feature points on each face
DATA_NUMS = 212  # 106 points (x ,y): 106*2 = 212

if __name__ == '__main__':
    # initialize the stream manager
    stream_manager = StreamManagerApi()
    stream_state = stream_manager.InitManager()
    if stream_state != 0:
        error_message = "Failed to init Stream manager, stream_state=%s" % str(stream_state)
        raise AssertionError(error_message)

    # create streams by pipeline config file
    with open("pipeline/faceswap.pipeline", 'rb') as f:
        pipeline = f.read().replace(b'\r', b'').replace(b'\n', b'')
    pipeline_string = pipeline

    stream_state = stream_manager.CreateMultipleStreams(pipeline_string)
    if stream_state != 0:
        error_message = "Failed to create Stream, streamState=%s" % str(stream_state)
        raise AssertionError(error_message)

    # prepare the input of the stream #begin

    # check the face img
    data_input1 = MxDataInput()
    data_input2 = MxDataInput()
    if os.path.exists(FACE1_PATH) != 1:
        error_message = 'The face1 image does not exist.'
        raise AssertionError(error_message)
    if os.path.exists(FACE2_PATH) != 1:
        error_message = 'The face2 image does not exist.'
        raise AssertionError(error_message)

    with open(FACE1_PATH, 'rb') as f1:
        data_input1.data = f1.read()

    with open(FACE2_PATH, 'rb') as f2:
        data_input2.data = f2.read()
    # prepare the input of the stream #end

    # send the prepared data to the stream
    data_input = [data_input1, data_input2]
    face_detect_info = []
    landmarks_infer_info = []
    count_picture = 1
    for i in data_input:
        # send the prepared data to the stream
        unique_id = stream_manager.SendData(STREAM_NAME, IN_PLUGIN_ID, i)
        if unique_id < 0:
            error_message = 'Failed to send data to stream.'
            raise AssertionError(error_message)

        # construct the resulted streamStateurned by the stream
        plugin_names = [b"mxpi_objectpostprocessor0",b"mxpi_tensorinfer1"]
        plugin_vector = StringVector()
        for plugin in plugin_names:
            plugin_vector.push_back(plugin)

        # get the output data according to the relevant plugins
        infer_result = stream_manager.GetProtobuf(STREAM_NAME, IN_PLUGIN_ID, plugin_vector)

        # checking whether the infer results is valid or not
        if infer_result.size() == 0:
            error_message = 'inferResult is null'
            raise IndexError(error_message)

        if infer_result[0].errorCode != 0:
            error_message = 'Unable to get effective infer results, please check the stream log for details'
            raise IndexError(error_message)

        # the output information of "mxpi_objectpostprocessor0"
        object_list = MxpiDataType.MxpiObjectList()
        object_list.ParseFromString(infer_result[0].messageBuf)

        # only select the image with a "face" label
        for item in object_list.objectVec:
            if item.classVec[0].className == "face":
                face_detect_info.append(object_list.objectVec[0])
            else:
                error_message = "There's no obvious face in the picture, please input pictures as required."
                raise AssertionError(error_message)

        # the output information of "mxpi_tensorinfer1" which is used to detect the features points of a crop face
        points_infer_list = MxpiDataType.MxpiTensorPackageList()
        points_infer_list.ParseFromString(infer_result[1].messageBuf)
        res = np.frombuffer(points_infer_list.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        landmarks_infer_info.append(res.tolist())

    # swap the cropped face # begin
    # get the landmark points of faces
    face1_points = landmarks_infer_info[0]
    face2_points = landmarks_infer_info[1]

    # the rectangular coordinate points of face detection #begin
    crop_face1_left = face_detect_info[0].x0
    crop_face1_right = face_detect_info[0].x1
    crop_face1_bottom = face_detect_info[0].y0
    crop_face1_top = face_detect_info[0].y1
    crop_face1_width = crop_face1_right - crop_face1_left
    crop_face1_height = crop_face1_top - crop_face1_bottom

    crop_face2_left = face_detect_info[1].x0
    crop_face2_right = face_detect_info[1].x1
    crop_face2_bottom = face_detect_info[1].y0
    crop_face2_top = face_detect_info[1].y1
    crop_face2_width = crop_face2_right - crop_face2_left
    crop_face2_height = crop_face2_top - crop_face2_bottom
    # the rectangular coordinate points of face detection #end

    #  get the cropped face #begin
    base_face = cv2.imread(FACE1_PATH)
    cover_face = cv2.imread(FACE2_PATH)
    crop_base_face = base_face[int(crop_face1_bottom): int(crop_face1_top), int(crop_face1_left): int(crop_face1_right)]
    crop_cover_face = cover_face[int(crop_face2_bottom): int(crop_face2_top), int(crop_face2_left):int(crop_face2_right)]
    #  get the cropped face #end

    for i in range(0, DATA_NUMS):
        if i % 2 ==0:
            face1_points[i] = int(face1_points[i] * crop_face1_width)
            face2_points[i] = int(face2_points[i] * crop_face2_width)
        else:
            face1_points[i] = int(face1_points[i] * crop_face1_height)
            face2_points[i] = int(face2_points[i] * crop_face2_height)
    base_points = np.array(face1_points, dtype=np.int32).reshape(POINTS_NUMS, 2)
    cover_points = np.array(face2_points, dtype=np.int32).reshape(POINTS_NUMS, 2)
    baseLandmarks = mat(base_points)
    coverLandmarks = mat(cover_points)

    # scan <faceswap_post.py> for more details of this process
    swap_face(baseLandmarks, coverLandmarks, crop_base_face, crop_cover_face)

    # swap the cropped face #end
    face_swap_result = cv2.imread("./only_face_swap.jpg")

    # merge the face_swap result into the source image
    base_face[int(crop_face1_bottom): int(crop_face1_top), int(crop_face1_left):int(crop_face1_right)] = face_swap_result
    cv2.imwrite("face_swap_result.jpg", base_face)

    # delete the intermediate process picture
    os.remove("./only_face_swap.jpg")

    # destroy streams
    stream_manager.DestroyAllStreams()
