#!/usr/bin/env python
# -*- encoding: utf-8 -*-
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

import io
import json
import os
import sys
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxProtobufIn, InProtobufVector, StringVector
import cv2
import numpy as np
from utils import preprocess , postprocess , valid


def evaluate(input_path, hr_path , stream_manager_api):
    print("Processing " + input_image_path + "...")
    if os.path.exists(input_image_path) != 1:
        print("The input image does not exist.")
        exit()
    
    tensor_data , origin_size = preprocess(input_path)
    tensor = tensor_data[None, :]

    vision_list = MxpiDataType.MxpiVisionList()
    vision_vec = vision_list .visionVec.add()
    vision_info =  vision_vec.visionInfo
    vision_info.width = 256
    vision_info.height = 256
    vision_info.widthAligned = 256
    vision_info.heightAligned = 256
    vision_data =  vision_vec.visionData
    vision_data.dataStr = tensor.tobytes()
    vision_data.deviceId = 0
    vision_data.memType = 0
    vision_data.dataSize = len(tensor)

    KEY0 = b"appsrc0"
    protobufVec = InProtobufVector()
    
    protobuf = MxProtobufIn()
    protobuf.key = KEY0
    protobuf.type = b"MxTools.MxpiVisionList"
    protobuf.protobuf = vision_list.SerializeToString()
    protobufVec.push_back(protobuf)

    
    STREAMNAME = b'superResolution'
    INPLUGINID = 0
    uniqueId = stream_manager_api.SendProtobuf(STREAMNAME, INPLUGINID, protobufVec)

    # get plugin output data
    key = b"mxpi_tensorinfer0"
    keyVec = StringVector()
    keyVec.push_back(key)
    inferResult = stream_manager_api.GetProtobuf(STREAMNAME, 0, keyVec)
    if inferResult.size() == 0:
        print("inferResult is null")
        exit()
    if inferResult[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
            inferResult[0].errorCode, inferResult[0].messageName.decode()))
        exit()

    # get the infer result
    inferList0 = MxpiDataType.MxpiTensorPackageList()
    inferList0.ParseFromString(inferResult[0].messageBuf)
    inferData = inferList0.tensorPackageVec[0].tensorVec[0].dataStr
    output = np.frombuffer(inferData, dtype=np.float32)

    # postprocess and valid
    hr = cv2.imread(hr_path)
    img = postprocess(output , hr.shape)

    result_path = "./result/" + str(input_path[-8:])
    cv2.imwrite(result_path , img)
    psnr_val = valid(img , hr)
    print("Infer finished.")
    return psnr_val

if __name__ == '__main__':
    
    # init stream manager
    StreamManagerApi = StreamManagerApi()
    ret = StreamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    pipeline = {
        "superResolution": {
            "stream_config": {
                "deviceId": "3"
            },
            "appsrc0": {
                "props": {
                    "blocksize": "409600"
                },
                "factory": "appsrc",
                "next": "mxpi_tensorinfer0"
            },
            "mxpi_tensorinfer0": {
                "props": {
                    "dataSource": "appsrc0",
                    "modelPath": "./model/srflow_df2k_x8_bs1.om"
                },
                "factory": "mxpi_tensorinfer",
                "next": "appsink0"
            },
            "appsink0": {
                "props": {
                    "blocksize": "409600"
                },
                "factory": "appsink"
            }
        }
    }
    pipelineStr = json.dumps(pipeline).encode()
    ret = StreamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    gt_set_path = "./dataset/div2k-validation-modcrop8-gt"
    x8_set_path = "./dataset/div2k-validation-modcrop8-x8"

    if os.path.exists(gt_set_path) != 1:
        print('The image set path {} does not exist.'.format(gt_set_path))
        exit()
    
    if os.path.exists(x8_set_path) != 1:
        print('The image set path {} does not exist.'.format(gt_set_path))
        exit()

    # get all image files
    image_files = os.listdir(x8_set_path)
    # sort by file name
    image_files.sort(key=lambda x: str(x[:-4]))
    
    num = 0
    pnsr_sum = 0
    for i in range(len(image_files)):
        input_image_path = os.path.join(x8_set_path , image_files[i])
        hr_image_path = os.path.join(gt_set_path , image_files[i])
        pnsr_sum += evaluate(input_image_path, hr_image_path , StreamManagerApi)
        num += 1

    print("Average pnsr value = " , pnsr_sum / num)
    StreamManagerApi.DestroyAllStreams()