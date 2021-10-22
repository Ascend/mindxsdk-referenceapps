#!/usr/bin/env python
# coding=utf-8
"""
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
"""

import sys
import json
import os
import cv2 as cv
import numpy as np

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector, MxProtobufIn, InProtobufVector


MODEL_WIDTH = 224
MODEL_HEIGHT = 224
OUT_W = 56
OUT_H = 56
CHANL_NUM = 2
IMG_CHW_NUM = 3
IMG_CHW_MAX = 255
L_CHANL_DATA = 50 
OUTPUT_DIR = '../out/'

def preprocess(picPath):
    # 抽取黑白图像L通道

    bgr_img = cv.imread(picPath).astype(np.float32)
    orig_shape = bgr_img.shape[:2]
    bgr_img = bgr_img / IMG_CHW_MAX
    lab_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2Lab)
    orig_l = lab_img[:, :, 0]
    if not orig_l.flags['C_CONTIGUOUS']:
        orig_l = np.ascontiguousarray(orig_l)
        
    lab_img = cv.resize(lab_img, (MODEL_WIDTH, MODEL_HEIGHT)).astype(np.float32)
    l_data = lab_img[:, :, 0]

    if not l_data.flags['C_CONTIGUOUS']:
        l_data = np.ascontiguousarray(l_data)

    l_data = l_data - L_CHANL_DATA

    return orig_shape, orig_l, l_data


def postprocess(result_list, pic, orig_shape, orig_l):
    # 预测a,b通道与初始L通道拼接获得彩色图像

    result_list = result_list.reshape(1, CHANL_NUM, OUT_W, OUT_H).transpose(0, CHANL_NUM, IMG_CHW_NUM, 1)
    result_array = result_list[0]

    ab_data = cv.resize(result_array, orig_shape[::-1])
    result_lab = np.concatenate((orig_l[:, :, np.newaxis], ab_data), axis=2)
    result_bgr = (IMG_CHW_MAX * np.clip(cv.cvtColor(result_lab, cv.COLOR_Lab2BGR), 0, 1)).astype('uint8')

    file_name = os.path.join(OUTPUT_DIR, "out_" + os.path.basename(pic))
    cv.imwrite(file_name, result_bgr)


if __name__ == '__main__':
    inputPic = sys.argv[1] 

    # 新建一个流管理StreamManager对象并初始化
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # 构建pipeline
    pipeline = b"../pipeline/colorization.pipeline" 
    ret = streamManagerApi.CreateMultipleStreamsFromFile(pipeline)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # 输入图片前处理 
    if os.path.exists(inputPic) != 1:
        print("The test image does not exist.")
    
    origShape, origL, lData = preprocess(inputPic)
    
    # 根据流名将检测目标传入流中
    streamName = b'colorization'
    inPluginId = 0

    tensor = lData[None, None, :]
    tensorPackageList = MxpiDataType.MxpiTensorPackageList()
    tensorPackage = tensorPackageList.tensorPackageVec.add()

    array_bytes = tensor.tobytes()
    dataInput = MxDataInput()
    dataInput.data = array_bytes
    tensorVec = tensorPackage.tensorVec.add()
    tensorVec.deviceId = 0
    tensorVec.memType = 0
    for i in tensor.shape:
        tensorVec.tensorShape.append(i)
    tensorVec.dataStr = dataInput.data
    tensorVec.tensorDataSize = len(array_bytes)
    
    key0 = b"appsrc0"
    protobufVec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key0
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensorPackageList.SerializeToString()
    protobufVec.push_back(protobuf)

    uniqueId = streamManagerApi.SendProtobuf(streamName, inPluginId, protobufVec)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    # 从流中取出数据
    key1 =  b"mxpi_tensorinfer0"
    keyVec = StringVector()
    keyVec.push_back(key1)
    
    inferRes = streamManagerApi.GetProtobuf(streamName, inPluginId, keyVec)

    if inferRes.size() == 0:
        print("inferResult is null")
        exit()
    if inferRes[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" % (
            inferRes[0].errorCode))
        exit()
    tensorList = MxpiDataType.MxpiTensorPackageList()
    tensorList.ParseFromString(inferRes[0].messageBuf)
    
    res = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
    
    # 推理结果后处理并输出结果
    postprocess(res, inputPic, origShape, origL)

    # 销毁流 
    streamManagerApi.DestroyAllStreams()
