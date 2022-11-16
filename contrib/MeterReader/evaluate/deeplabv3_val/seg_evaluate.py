# !/usr/bin/env python
# coding=utf-8

# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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

import json
import os
import stat
import cv2
import numpy as np

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

pipeline_path = '../pipeline/deeplabv3/seg.pipeline'
DIRNAME = "../evaluate/deeplabv3_val/seg_val_img/seg_test_img"
det_val_dir = "../evaluate/deeplabv3_val/seg_val_img/seg_test_img_groundtruth"

def miou_computer(miuo_img, miou_pred):
    """
    miuo_img: 已标注的原图片
    img_pred: 预测出的图片
    """
    if (miuo_img.shape != miou_pred.shape):
        print("两个图片形状不一致")
        exit()

    unique_item_list = np.unique(miuo_img)  
    unique_item_dict = {} 
    for index in unique_item_list:
        item = index
        unique_item_dict[item] = index
    num = len(np.unique(unique_item_list))  

    # 混淆矩阵
    _m = np.zeros((num + 1, num + 1)) 
    for i in range(miuo_img.shape[0]):
        for j in range(miuo_img.shape[1]):
            _mi = unique_item_dict.get(miuo_img[i][j])
            _mj = unique_item_dict.get(miou_pred[i][j])
            _m[_mi][_mj] += 1
    # 前num行相加，存在num+1列 【实际下标-1】
    _m[:num, num] = np.sum(_m[:num, :num], axis=1)
    # 前num+1列相加，放在num+1行【实际下标-1】
    _m[num, :num + 1] = np.sum(_m[:num, :num + 1], axis=0)

    # 计算Miou值
    miou = 0
    for i in range(num):
        miou += (_m[i][i]) / (_m[i][num] + _m[num][i] - _m[i][i])
    miou /= num
    return miou


if __name__ == '__main__':
    steammanager_api = StreamManagerApi()
    # init stream manager
    ret = steammanager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    MODES = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(pipeline_path, os.O_RDONLY, MODES), 'rb') as f:
        pipeline_str = f.read()
    ret = steammanager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    dataInput = MxDataInput()
    # It is best to use absolute path
    # 语义分割模型推理
    files = os.listdir(DIRNAME)
    det_pred_dict = {}
    for file in files:
        FILENAME = DIRNAME + os.path.sep + file
        print(file)
        if os.path.exists(FILENAME) != 1:
            print("The test image does not exist. Exit.")
            exit()
        with os.fdopen(os.open(FILENAME, os.O_RDONLY, MODES), 'rb') as f:
            dataInput.data = f.read()
        STEAMNAME = b'seg'
        INPLUGINID = 0
        uniqueId = steammanager_api.SendData(STEAMNAME, INPLUGINID, dataInput)
        if uniqueId < 0:
            print("Failed to send data to stream.")
            exit()
        keys = [b"mxpi_semanticsegpostprocessor0"]
        keyVec = StringVector()
        for key in keys:
            keyVec.push_back(key)
        # ### Test

        infer = steammanager_api.GetResult(STEAMNAME, b'appsink0', keyVec)
        if (infer.metadataVec.size() == 0):
            print("Get no data from stream !")
            exit()
        infer_result = infer.metadataVec[0]
        if infer_result.errorCode != 0:
            print("GetResult error. errorCode=%d , errMsg=%s" % (infer_result.errorCode, infer_result.errMsg))
            exit()
        result = MxpiDataType.MxpiImageMaskList()
        result.ParseFromString(infer_result.serializedMetadata)
        pred = np.frombuffer(result.imageMaskVec[0].dataStr
                             , dtype=np.uint8)
        img_pred = pred.reshape((512, 512))

        det_pred_dict[file.split(".")[0]] = img_pred

    # 获取标注数据
    MODES = stat.S_IWUSR | stat.S_IRUSR
    files = os.listdir(det_val_dir)
    det_val_dict = {}
    for file in files:
        FILENAME = det_val_dir + os.path.sep + file

        if os.path.exists(FILENAME) != 1:
            print("The test image does not exist. Exit.")
            exit()
        temp = cv2.imread(FILENAME, 0)
        img = cv2.resize(temp, (512, 512))
        det_val_dict[file.split(".")[0]] = img

    unique_keys = set(det_pred_dict.keys())
    unique_keys.intersection(set(det_val_dict.keys()))
    _unique = list(unique_keys)
    Miou_list = []
    for key in _unique:
        Miou_list.append(miou_computer(det_val_dict.get(key), det_pred_dict.get(key)))
    print("The average miou is=====================:", np.average(np.array(Miou_list)))

    steammanager_api.DestroyAllStreams()
