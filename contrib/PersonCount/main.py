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
import cv2
import scipy.io as sio
import numpy as np
import math
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
import MxpiDataType_pb2 as MxpiDataType
import time

if __name__ == '__main__':
    # create streams
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    #pipeline config include several types of plugins.
    #input plugin
    #multi-media picture preprocess plugin
    #tensorinfer plugin
    #postprocess plugin
    #output plugin
    pipeline = {
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
    pipelineStr = json.dumps(pipeline).encode()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    # Construct the input of the stream
    dataInput = MxDataInput()
    #dataset path needs to fixde as specific path
    Dataset_Path = '../tools/python_count/ShanghaiTech/part_B_images/'
    namelist = os.listdir(Dataset_Path)
    person_num_list=[]
    gt_list=[]
    #start time
    time_start=time.time()
    uniqueIds=[]
    streamName = b'detection'
    #the shape of output image and output heat map
    image_H=800
    image_W=1408
    #the pixel position of the person number text embedded in the heat map
    position=300
    #infer all the picture in target Dataset_Path directory
    for i in range(1, len(namelist) + 1):
        with open(Dataset_Path + 'IMG_' + str(i) + '.jpg', 'rb') as f:
            data = f.read()
        inPluginId = i
        dataInput.data = data
        # Inputs data to a specified stream based on streamName.
        # continuous datasend is used to support batch mechanism
        uniqueId = streamManagerApi.SendData(streamName, 0, dataInput)
        if uniqueId < 0:
            print("Failed to send data to stream.")
            exit()
        uniqueIds.append(uniqueId)
    key = b"mxpi_objectpostprocessor0"
    keyVec = StringVector()
    keyVec.push_back(key)
    for i in range(1, len(namelist)+1):
        # Obtain the inference result by specifying streamName and uniqueId.
        infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)
        objectList = MxpiDataType.MxpiObjectList()
        objectList.ParseFromString(infer_result[0].messageBuf)
        results = objectList.objectVec[0]
        #the persor num is stored in classId attribution.
        person_num=results.classVec[0].classId
        person_num_list.append(person_num)
        #output heatmap is stored in mask attribution.
        data=results.imageMask.dataStr
        data=np.frombuffer(data, dtype=np.uint8)
        #the data is reshape as origin image size
        data=data.reshape((image_H, image_W))
        image=cv2.applyColorMap(data, cv2.COLORMAP_JET)
        #person num txt is embedded into heatmap.
        text="Count: " + str(person_num)
        RGB=(0, 0, 255)
        cv2.putText(image, text, (position, position), cv2.FONT_HERSHEY_SIMPLEX, 4, RGB, 4)
        cv2.imwrite("./heat_map/" + str(i) + "_heatmap.jpg", image)
        #load ground truth information
        #gt_num represents the person number of ground truth.
        #the ground truth needs to fix as specific path.
        gt_path="../tools/python_count/ShanghaiTech/part_B_test/GT_IMG_" + str(i) + ".mat"
        data1=sio.loadmat(gt_path)
        gt_num=int(data1['image_info'][0][0][0][0][1][0][0])
        gt_list.append(gt_num)
    #end time
    time_end = time.time()
    print("total image number:", len(namelist))
    print('time cost', time_end - time_start, 's')
    MAE = 0
    MSE = 0
    #computing mse of prediction value and ground truth
    for i in range(len(namelist)):
        MAE += abs(person_num_list[i] - gt_list[i])
        MSE += (person_num_list[i] - gt_list[i]) ** 2
    MAE /= len(namelist)
    MSE = (MSE / len(namelist)) ** 0.5
    print("MAE:", MAE, "\tMSE:", MSE)
    # destroy streams
    streamManagerApi.DestroyAllStreams()
