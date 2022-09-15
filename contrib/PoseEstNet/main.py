#!/usr/bin/env python
# coding=utf-8

"""
Copyright(C) Huawei Technologies Co.,Ltd. 2012-2022 All rights reserved.

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

import argparse
import os
import cv2

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

YUV_BYTES_NU = 3
YUV_BYTES_DE = 2

POSTESTNET_STREAM_NAME = b'PoseEstNetProcess'
IN_PLUGIN_ID = 0


def initialize_stream():
    """
    Initialize stream

    :arg:
        None

    :return:
        Stream api
    """
    stream_api = StreamManagerApi()
    stream_state = stream_api.InitManager()
    if stream_state != 0:
        error_message = "Failed to init Stream manager, stream_state=%s" % str(stream_state)
        print(error_message)
        exit()

    # creating stream based on json strings in the pipeline file: 'PoseEstNet.pipeline'
    with open("pipeline/PoseEstNet.pipeline", 'rb') as f:
        pipeline = f.read().replace(b'\r', b'').replace(b'\n', b'')
    pipeline_string = pipeline

    stream_state = stream_api.CreateMultipleStreams(pipeline_string)
    if stream_state != 0:
        error_message = "Failed to create Stream, stream_state=%s" % str(stream_state)
        print(error_message)
        exit()

    return stream_api


def process(input_path, stream_api):
    """
    :arg:
        queryPath: the directory of query images
        streamApi: stream api

    :return:
        queryFeatures: the vectors of queryFeatures
        queryPid: the vectors of queryPid
    """

    # constructing the results returned by the queryImageProcess stream
    plugin_names = [b"mxpi_distributor0_0", b"mxpi_postprocess1"]

    plugin_name_vector = StringVector()
    for key in plugin_names:
        plugin_name_vector.push_back(key)

    # check the query file
    if os.path.exists(input_path) != 1:
        error_message = 'The file of input images does not exist.'
        print(error_message)
        exit()
    if len(os.listdir(input_path)) == 0:
        error_message = 'The file of input images is empty.'
        print(error_message)
        exit()

    # extract the features for all images in query file
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if not file.endswith('.jpg'):
                print('Input image only support jpg')
                exit()

            query_data_input = MxDataInput()
            file_path = os.path.join(root, file)
            with open(file_path, 'rb') as f:
                query_data_input.data = f.read()

            # send the prepared data to the stream
            unique_id = stream_api.SendData(POSTESTNET_STREAM_NAME, IN_PLUGIN_ID, query_data_input)
            if unique_id < 0:
                error_message = 'Failed to send data to queryImageProcess stream.'
                print(error_message)
                exit()

            # get infer result
            infer_result = stream_api.GetProtobuf(POSTESTNET_STREAM_NAME, IN_PLUGIN_ID, plugin_name_vector)

            # checking whether the infer results is valid or not
            if infer_result.size() == 0:
                error_message = 'Unable to get effective infer results, please check the stream log for details'
                print(error_message)
                exit()
            if infer_result[0].errorCode != 0:
                error_message = "GetProtobuf error. errorCode=%d, error_message=%s" % (infer_result[0].errorCode,
                                                                                     infer_result[0].messageName)
                print(error_message)
                exit()

            # get the output information of "mxpi_objectpostprocessor0" plugin
            car_object_list = MxpiDataType.MxpiObjectList()
            car_object_list.ParseFromString(infer_result[0].messageBuf)
            car_num = len(car_object_list.objectVec)

            # get the output information of "mxpi_objectpostprocessor0" plugin
            keypoint_object_list = MxpiDataType.MxpiObjectList()
            keypoint_object_list.ParseFromString(infer_result[1].messageBuf)
            keypoint_num = len(keypoint_object_list.objectVec)
            if keypoint_num // car_num != 36:
                error_message = 'Failed to map the inferred key points to the detected cars.'
                print(error_message)
                exit()

            original_img = cv2.imread(file_path)
            for index in range(len(keypoint_object_list.objectVec)):
                original_x = int(car_object_list.objectVec[index // 36].x0) + keypoint_object_list.objectVec[index].x0
                original_y = int(car_object_list.objectVec[index // 36].y0) + keypoint_object_list.objectVec[index].y0
                visible = keypoint_object_list.objectVec[index].x1
                if round(visible):
                    cv2.circle(original_img, (int(original_x), int(original_y)), 2, [255, 0, 0], 2)
            cv2.imwrite("output/result_{}".format(str(file)), original_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputPath', type=str, default='data')
    opt = parser.parse_args()
    streamManagerApi = initialize_stream()
    process(opt.inputPath, streamManagerApi)