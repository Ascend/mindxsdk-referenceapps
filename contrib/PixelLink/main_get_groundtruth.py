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


import MxpiDataType_pb2 as MxpiDataType
import numpy as np
import cv2
import process2
import sys
sys.path.append("../../proto")
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector


if __name__ == '__main__':
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))

    # create streams by pipeline config file
    with open("Pixel.pipeline", 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
    # the number of the image is 500
    for img_id in range(1, 501):
        # Construct the input of the stream
        data_input = MxDataInput()
        with open("./ch4_test_images/" + "img_" + str(img_id) + ".jpg", 'rb') as f:
            data_input.data = f.read()
        print("Now, we are dealing the number:", img_id)
        # Inputs data to a specified stream based on streamName.
        stream_name = b'classification'
        inPlugin_id = 0
        unique_id = stream_manager_api.SendData(stream_name, inPlugin_id, data_input)
        if unique_id < 0:
            print("Failed to send data to stream.")
        # get protobuf with custom
        key_vec = StringVector()
        # choose which metadata to be got.In this case we use the custom "mxpi_sampleproto"
        key_vec.push_back(b"pixelLink_process")
        # get inference result
        infer = stream_manager_api.GetResult(stream_name, b'appsink0', key_vec)
        infer_result = infer.metadataVec[0]
        if infer_result.errorCode != 0:
            print("GetResult error. errorCode=%d ,errorMsg=%s" % (
                infer_result.errorCode, infer_result.errorMsg))
            exit()
        tensorList = MxpiDataType.MxpiTensorPackageList()
        tensorList.ParseFromString(infer_result.serializedMetadata)

        ids = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        ids2 = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[1].dataStr, dtype=np.float32)
        shape = tensorList.tensorPackageVec[0].tensorVec[0].tensorShape
        ids.resize(shape)
        # if batch exsit, shape need remove dim0
        shape2 = tensorList.tensorPackageVec[0].tensorVec[1].tensorShape
        ids2.resize(shape2)
        np.set_printoptions(threshold=sys.maxsize)

        # post-processing
        img_height = 192
        img_width = 320
        result = np.zeros((1, img_height, img_width), dtype=float)
        for i in range(img_height):
            for j in range(img_width):
                num1 = np.exp(ids[0][i][j][0])
                num2 = np.exp(ids[0][i][j][1])
                max1 = num1 / (num1 + num2)
                max2 = num2 / (num1 + num2)
                result[0][i][j] = max2
        result2 = np.zeros((1, img_height, img_width, 8), dtype=float)
        for i in range(img_height):
            for j in range(img_width):
                for k in range(0, 16, 2):
                    num1 = np.exp(ids2[0][i][j][k])
                    num2 = np.exp(ids2[0][i][j][k + 1])
                    max1 = num1 / (num1 + num2)
                    max2 = num2 / (num1 + num2)
                    if k < 2:
                        result2[0][i][j][k] = max2
                    else:
                        result2[0][i][j][k // 2] = max2
        process2.deal(result, result2, img_id)
        print("done the number of image is : ", img_id)
