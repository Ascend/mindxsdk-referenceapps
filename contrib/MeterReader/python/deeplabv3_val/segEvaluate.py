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


def miouComputer(img, img_pred):
    """
    img: 已标注的原图片
    img_pred: 预测出的图片
    """
    if (img.shape != img_pred.shape):
        print("两个图片形状不一致")
        return

    unique_item_list = np.unique(img)  
    unique_item_dict = {} 
    for index in unique_item_list:
        item = index
        unique_item_dict[item] = index
    num = len(np.unique(unique_item_list))  

    # 混淆矩阵
    M = np.zeros((num + 1, num + 1)) 
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            MI = unique_item_dict.get(img[i][j])
            MJ = unique_item_dict.get(img_pred[i][j])
            M[MI][MJ] += 1
    # 前num行相加，存在num+1列 【实际下标-1】
    M[:num, num] = np.sum(M[:num, :num], axis=1)
    # 前num+1列相加，放在num+1行【实际下标-1】
    M[num, :num + 1] = np.sum(M[:num, :num + 1], axis=0)

    # 计算Miou值
    miou = 0
    for i in range(num):
        miou += (M[i][i]) / (M[i][num] + M[num][i] - M[i][i])
    miou /= num
    return miou


if __name__ == '__main__':
    # 获取当前脚本的位置

    cur_path = os.path.abspath(os.path.dirname(__file__))
    father_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    model_path = os.path.join(father_path, 'models', 'deeplabv3', 'seg.om').replace('\\', '/')
    postProcessConfigPath = os.path.join(father_path, 'pipeline', 'deeplabv3', 'deeplabv3.cfg').replace('\\', '/')
    labelPath = os.path.join(father_path, 'pipeline', 'deeplabv3', 'deeplabv3.names').replace('\\', '/')
    pipeline_path = os.path.join(father_path, 'pipeline', 'deeplabv3', 'seg.pipeline').replace('\\', '/')
    DIRNAME = os.path.join(cur_path, 'seg_val_img', 'seg_test_img').replace('\\', '/')
    det_val_dir = os.path.join(cur_path, 'seg_val_img', 'seg_test_img_groundtruth').replace('\\', '/')

    # 改写pipeline里面的model路径

    file_object = open(pipeline_path, 'r')

    content = json.load(file_object)
    modelPath = model_path
    content['seg']['mxpi_tensorinfer0']['props']['modelPath'] = modelPath
    content['seg']['mxpi_semanticsegpostprocessor0']['props']['postProcessConfigPath'] = postProcessConfigPath
    content['seg']['mxpi_semanticsegpostprocessor0']['props']['labelPath'] = labelPath

    with os.fdopen(os.open(pipeline_path, 'os.O_WRONLY', 'stat.S_IWUSR'), 'w') as f:
        json.dump(content, f)

    

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
    unique_keys = list(unique_keys)
    Miou_list = []
    for key in unique_keys:
        Miou_list.append(miouComputer(det_val_dict.get(key), det_pred_dict.get(key)))
    print(np.average(np.array(Miou_list)))

    steammanager_api.DestroyAllStreams()
