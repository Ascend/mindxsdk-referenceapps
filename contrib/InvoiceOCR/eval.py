#!/usr/bin/env python
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


import os
import sys
import copy
import math
import json
import time
import cv2
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector


globals_convert = {
    'true': 0,
    'false': 1
}


def get_file():
    fs = []
    gts = []
    with open('./eval_data/Label_test.txt', 'r', encoding='utf-8') as file:
        label_list = [line.strip('\n') for line in file]
    for item in label_list:
        file_name, label = item.split('\t')
        fs.append(file_name)
        gts.append(eval(label, globals_convert))
    return fs, gts


# iou
def iou(polygon_1, polygon_2):
    x_min = min(polygon_1[0][0], polygon_1[3][0], polygon_2[0][0], polygon_2[3][0])
    x_max = max(polygon_1[1][0], polygon_1[2][0], polygon_2[1][0], polygon_2[2][0])
    y_min = min(polygon_1[0][1], polygon_1[1][1], polygon_2[0][1], polygon_2[1][1])
    y_max = max(polygon_1[2][1], polygon_1[3][1], polygon_2[2][1], polygon_2[3][1])
    a = np.zeros((y_max, x_max), np.uint8)
    b = np.zeros((y_max, x_max), np.uint8)
    cv2.fillConvexPoly(a, np.array(polygon_1), 255)
    cv2.fillConvexPoly(b, np.array(polygon_2), 255)
    c = cv2.bitwise_and(a, b)
    a_area = cv2.countNonZero(a)
    b_area = cv2.countNonZero(b)
    c_area = cv2.countNonZero(c)
    iou_p = c_area / (a_area + b_area - c_area)
    return iou_p


def str_contrast(str1, str2):
    s1 = list(str1)
    s2 = list(str2)
    count = 0
    for s in s1:
        if s in s2:
            s2.remove(s)
            count += 1
    return count


def eval_value(pre, gt):
    text_sum = 0
    for item in gt:
        text_sum += len(item["transcription"])

    correct_sum = 0
    for points in pre:
        pre_box = [[points['x0'], points['y0']], [points['x1'], points['y1']], [points['x2'], points['y2']],
                   [points['x3'], points['y3']]]
        pre_text = points['MxpiTextsInfo'][0]['text'][0]
        fit = 0
        fit_index = -1
        for j, gt_p in enumerate(gt):
            gt_box = gt_p['points']
            iou_p = iou(pre_box, gt_box)
            if iou_p > 0.3 and iou_p > fit:
                fit_index = j
                fit = iou_p
        if fit_index >= 0:
            gt_text = gt[fit_index]['transcription']
            correct_sum += str_contrast(pre_text, gt_text)
    value = correct_sum / text_sum
    return value


if __name__ == '__main__':

    test_files, gts_list = get_file()

    if len(test_files) == 0:
        print("The eval directory is EMPTY!")
        print("Please place the picture in './eval_data/' !")
        exit()

    # 创建流管理对象
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # 读取pipeline
    with open("./pipeline/InvoiceOCR.pipeline", 'rb') as f:
        pipeline = f.read().replace(b'\r', b'').replace(b'\n', b'')
    pipelineStr = pipeline

    # 创建流
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    acc_sum = int(0)
    start = time.time()
    for i, test_file in enumerate(test_files):

        filename = test_file
        gt_list = gts_list[i]
        # 添加路径
        testfile = "eval_data/" + filename

        if os.path.getsize(testfile) == 0:
            print("Error!The eval image is empty.")
            continue

        # 创建输入对象
        dataInput = MxDataInput()

        with open(testfile, 'rb') as f:
            dataInput.data = f.read()

        # 流信息
        STREAM_NAME = b'invoiceocr'
        INPLUGIN_ID = 0
        # 发送数据
        uniqueId = streamManagerApi.SendData(STREAM_NAME, INPLUGIN_ID, dataInput)

        inferResult = streamManagerApi.GetResult(STREAM_NAME, uniqueId, 3000000)
        end = time.time()
        total_time = []
        total_time.append(end - start)
        print('Running time: %s Seconds' % sum(total_time))
        print('===============================')
        if inferResult.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                inferResult.errorCode, inferResult.data.decode()))
            exit()
        result = json.loads(inferResult.data.decode())
        pre_list = result['MxpiTextObject']
        acc_sum += eval_value(pre_list, gt_list)
    acc = acc_sum / len(test_files)
    print("acc: ", acc)

    streamManagerApi.DestroyAllStreams()
