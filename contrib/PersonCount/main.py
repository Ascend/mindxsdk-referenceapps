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
import math
import time
import cv2
import scipy.io as sio
import numpy as np
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
import MxpiDataType_pb2 as MxpiDataType


if __name__ == '__main__':
    # create streams
    STREAM = StreamManagerApi()
    RET = STREAM.InitManager()
    if RET != 0:
        print("Failed to init Stream manager, RET=%s" % str(RET))
        exit()
    #pipeline config include several types of plugins.
    #input plugin
    #multi-media picture preprocess plugin
    #tensorinfer plugin
    #postprocess plugin
    #output plugin
    PIPELINE = {
        "detection": {
            "stream_config": {
                "deviceId": "0"
            },
            "appsrc0": {
                "props": {
                    "blocksize": "409600"
                },
                "factory": "appsrc",
                "next": "mxpi_imagedecoder0"
            },
            "mxpi_imagedecoder0": {
                "factory": "mxpi_imagedecoder",
                "next": "mxpi_imageresize0"
            },
            "mxpi_imageresize0": {
                "props": {
                  "resizeHeight":"800",
                  "resizeWidth":"1408",
                  "dataSource":"mxpi_imagedecoder0"
                },
                "factory": "mxpi_imageresize",
                "next": "mxpi_tensorinfer0"
            },
            "mxpi_tensorinfer0": {
                "props": {
                    "dataSource":"mxpi_imageresize0",
                    "modelPath": "model/count_person_8.caffe.om",
                },
                "factory": "mxpi_tensorinfer",
                "next": "mxpi_objectpostprocessor0"
            },
            "mxpi_objectpostprocessor0": {
                "props": {
                        "dataSource": "mxpi_tensorinfer0",
                        "funcLanguage":"c++",
                        "postProcessConfigPath": "config/insert_op.cfg",
                        "labelPath": "config/person.names",
                        "postProcessLibPath": "Plugin1/build/libcountpersonpostprocess.so",
                },
                "factory": "mxpi_objectpostprocessor",
                "next": "appsink0"
            },
            "appsink0": {
                "props": {
                    "blocksize": "40960000"
                },
                "factory": "appsink"
            }
        }
    }
    #transfer pipeline string into json format
    PIPELINESTR = json.dumps(PIPELINE).encode()
    RET = STREAM.CreateMultipleStreams(PIPELINESTR)
    if RET != 0:
        print("Failed to create Stream, RET=%s" % str(RET))
        exit()
    # Construct the input of the stream
    DATA_INPUT = MxDataInput()
    #dataset path needs to fixde as specific path
    DATASET_PATH = 'ShanghaiTech/part_B_images/'
    NAME_LIST = os.listdir(DATASET_PATH)
    PERSON_NUM_LIST = []
    GT_LIST = []
    #start time
    TIME_START = time.time()
    UNIQUEIDS = []
    STREAMNAME = b'detection'
    #the shape of output image and output heat map
    IMAGE_H = 800
    IMAGE_W = 1408
    #the pixel position of the person number text embedded in the heat map
    POSITION = 300
    #infer all the picture in target Dataset_Path directory
    for i in range(1, len(NAME_LIST) + 1):
        with open(DATASET_PATH + 'IMG_' + str(i) + '.jpg', 'rb') as f:
            data = f.read()
        inPluginId = i
        DATA_INPUT.data = data
        # Inputs data to a specified stream based on streamname.
        # continuous datasend is used to support batch mechanism
        uniqueId = STREAM.SendData(STREAMNAME, 0, DATA_INPUT)
        if uniqueId < 0:
            print("Failed to send data to stream.")
            exit()
        UNIQUEIDS.append(uniqueId)
    KEY = b"mxpi_objectpostprocessor0"
    KEYVEC = StringVector()
    KEYVEC.push_back(KEY)
    for i in range(1, len(NAME_LIST)+1):
        # Obtain the inference result by specifying streamname and uniqueId.
        infer_result = STREAM.GetProtobuf(STREAMNAME, 0, KEYVEC)
        objectList = MxpiDataType.MxpiObjectList()
        objectList.ParseFromString(infer_result[0].messageBuf)
        results = objectList.objectVec[0]
        #the persor num is stored in classId attribution.
        person_num = results.classVec[0].classId
        PERSON_NUM_LIST.append(person_num)
        #output heatmap is stored in mask attribution.
        data = results.imageMask.dataStr
        data = np.frombuffer(data, dtype=np.uint8)
        #the data is reshape as origin image size
        data = data.reshape((IMAGE_H, IMAGE_W))
        image = cv2.applyColorMap(data, cv2.COLORMAP_JET)
        #person num txt is embedded into heatmap.
        text = "Count: " + str(person_num)
        RGB = (0, 0, 255)
        cv2.putText(image, text, (POSITION, POSITION), cv2.FONT_HERSHEY_SIMPLEX, 4, RGB, 4)
        cv2.imwrite("./heat_map/" + str(i) + "_heatmap.jpg", image)
        #load ground truth information
        #gt_num represents the person number of ground truth.
        #the ground truth needs to fix as specific path.
        gt_path = "ShanghaiTech/part_B_test/GT_IMG_" + str(i) + ".mat"
        data1 = sio.loadmat(gt_path)
        gt_num = int(data1['image_info'][0][0][0][0][1][0][0])
        GT_LIST.append(gt_num)
    #end time
    TIME_END = time.time()
    print("total image number:", len(NAME_LIST))
    print('time cost', TIME_END - TIME_START, 's')
    MAE = 0
    MSE = 0
    #computing mse of prediction value and ground truth
    for i in range(len(NAME_LIST)):
        MAE += abs(PERSON_NUM_LIST[i] - GT_LIST[i])
        MSE += (PERSON_NUM_LIST[i] - GT_LIST[i]) ** 2
    MAE /= len(NAME_LIST)
    MSE = (MSE / len(NAME_LIST)) ** 0.5
    print("MAE:", MAE, "\tMSE:", MSE)
    # destroy streams
    STREAM.DestroyAllStreams()
