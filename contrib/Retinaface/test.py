#!/usr/bin/env python
# coding=utf-8

"""
Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.

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
import stat
import argparse
import json
import shutil
import tqdm
import cv2
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from PIL import Image
from utils import preprocess, postprocess
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
from StreamManagerApi import StreamManagerApi, MxProtobufIn, InProtobufVector, StringVector

RNDB = "./widerface/val/images/"
RNDY = "./evaluate/widerface_txt"
img_addresses = []
streamManagerApi = StreamManagerApi()
ret = streamManagerApi.InitManager()
if ret != 0:
    print("Failed to init Stream manager, ret=%s" % str(ret))
    exit()

pipeline = {
    "Retinaface": {
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
                "singleBatchInfer": "1",
                "dataSource": "appsrc0",
                "modelPath": "./model/newRetinaface.om"
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
ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
if ret != 0:
    print("Failed to create Stream, ret=%s" % str(ret))
    exit()

FLAGS = os.O_RDWR | os.O_CREAT
MODES = stat.S_IWUSR | stat.S_IRUSR
with os.fdopen(os.open('./evaluate/wider_val.txt', FLAGS, MODES), 'r') as fr:
    for img_address in fr:
        print(img_address)
        tensor_data , info = preprocess(RNDB + img_address.strip('\r\n'))
        tensor = tensor_data[None, :]
        STREAMNAME = b"Retinaface"
        INPLUGINID = 0
        visionList = MxpiDataType.MxpiVisionList()
        visionVec = visionList.visionVec.add()
        visionInfo = visionVec.visionInfo

        visionInfo.width = 1000
        visionInfo.height = 1000
        visionInfo.widthAligned = 1000
        visionInfo.heightAligned = 1000
        visionData = visionVec.visionData
        visionData.dataStr = tensor.tobytes()
        visionData.deviceId = 0
        visionData.memType = 0
        visionData.dataSize = len(tensor)

        KEY_VALUE = b"appsrc0"
        protobufVec = InProtobufVector()

        protobuf = MxProtobufIn()
        protobuf.key = KEY_VALUE
        protobuf.type = b"MxTools.MxpiVisionList"
        protobuf.protobuf = visionList.SerializeToString()
        protobufVec.push_back(protobuf)

        uniqueId = streamManagerApi.SendProtobuf(STREAMNAME, INPLUGINID, protobufVec)
        KEY_VALUE = b"mxpi_tensorinfer0"
        keyVec = StringVector()
        keyVec.push_back(KEY_VALUE)
        inferResult = streamManagerApi.GetProtobuf(STREAMNAME, 0, keyVec)

        infer_list = MxpiDataType.MxpiTensorPackageList()
        infer_list.ParseFromString(inferResult[0].messageBuf)
        infer_data0 = infer_list.tensorPackageVec[0].tensorVec[0].dataStr
        loc = np.frombuffer(infer_data0, dtype=np.float32)
        infer_data1 = infer_list.tensorPackageVec[0].tensorVec[1].dataStr
        conf = np.frombuffer(infer_data1, dtype=np.float32)
        infer_data2 = infer_list.tensorPackageVec[0].tensorVec[2].dataStr
        landms = np.frombuffer(infer_data2, dtype=np.float32)
        result , count = postprocess(loc , conf , landms , info)
        dir_name = RNDY + "/" + os.path.split(img_address.strip('.jpg\r\n'))[0]
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        print(dir_name)
        txt_name = RNDY + "/" + img_address.strip('.jpg\r\n') + '.txt'
        res_name = os.path.split(img_address.strip('.jpg\r\n'))[1] + "\n"
        with os.fdopen(os.open(txt_name, FLAGS, MODES), 'w') as f:
            f.write(res_name)
            f.write('{:d}\n'.format(count))
            f.write(result)
        f.close()

streamManagerApi.DestroyAllStreams()

