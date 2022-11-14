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
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector


def get_file_names(root_dir):
    fs = []
    for root, dirs, files in os.walk(root_dir, topdown=True):
        for name in files:
            _, ending = os.path.splitext(name)
            if ending == ".jpg":
                fs.append(os.path.join(name))

            if ending == ".JPG":
                # remove '.JPG' add '.jpg'
                name = name[:-4] + '.jpg'
                new_name = root_dir + name
                old_name = root_dir + name[:-4] + '.JPG'

                while os.path.exists(new_name):
                    new_name = root_dir + name[:-4] + '(1).jpg'
                    name = name[:-4] + '(1).jpg'
                os.rename(old_name, new_name)

                fs.append(os.path.join(name))
    return fs


def add_text(img, text, left, top, textColor, textSize):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontstyle = ImageFont.truetype("SIMSUN.TTC", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontstyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == '__main__':

    test_files = get_file_names('./inputs/')

    if len(test_files) == 0:
        print("The input directory is EMPTY!")
        print("Please place the picture in './inputs/' !")
        exit()

    # 创建并初始化流管理对象
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # 读取pipeline文件
    with open("./pipeline/InvoiceOCR.pipeline", 'rb') as f:
        pipeline = f.read().replace(b'\r', b'').replace(b'\n', b'')
    pipelineStr = pipeline

    # 创建流
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    if os.path.exists('outputs') is not True:
        os.mkdir('outputs')

    for testfile in test_files:
        filename = testfile

        # 添加路径
        testfile = "inputs/" + testfile

        if os.path.getsize(testfile) == 0:
            print("Error!The test image is empty.")
            continue

        # 创建输入对象
        dataInput = MxDataInput()

        with open(testfile, 'rb') as f:
            dataInput.data = f.read()

        # 流信息
        STREAM_NAME = b'invoiceocr'
        INPLUGIN_ID = 0
        # 发送输入至流
        uniqueId = streamManagerApi.SendData(STREAM_NAME, INPLUGIN_ID, dataInput)

        # 获取输出
        inferResult = streamManagerApi.GetResult(STREAM_NAME, uniqueId, 3000000)
        if inferResult.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                inferResult.errorCode, inferResult.data.decode()))
            exit()

        # 打印结果
        print()
        print("------------", filename, "output-start------------")
        print()
        result = json.loads(inferResult.data.decode())
        print(json.dumps(result, sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))
        print()
        print("------------", filename, "output-end------------")
        print()

        class_r = result["MxpiClass"]
        content = result["MxpiTextObject"]

        class_r = sorted(class_r, key=lambda x: -x['confidence'])
        class_id = class_r[0]['className']

        text = ''
        for item in content:
            text += (item['MxpiTextsInfo'][0]['text'][0])

        if '车号' in text or '证号' in text or '上车' in text or '下车' in text:
            if class_id != 'taxi_receipt':
                class_id = 'taxi_receipt'
        elif '增值税' in text or '纳税人识别号' in text:
            if class_id != 'vat_invoice':
                class_id = 'vat_invoice'
        else:
            if class_id != 'quota_invoice':
                class_id = 'quota_invoice'

        # 绘图参数
        BOX_COLOR = (255, 0, 0)
        TEXT_COLOR = (255, 0, 0)
        BOX_THICKNESS = 2

        img = cv2.imread(testfile)
        img_c = img.copy()
        for item in content:

            if item['MxpiTextsInfo'][0]['text'][0] == '':
                continue
            box = np.array([[item['x0'], item['y0']], [item['x1'], item['y1']], [item['x2'], item['y2']],
                            [item['x3'], item['y3']]])
            text_size = int((box[3][1] - box[0][1]) / 2)
            cv2.polylines(img_c, [box], True, BOX_COLOR, BOX_THICKNESS)
            img_c = add_text(img_c, item['MxpiTextsInfo'][0]['text'][0], item['x0'], item['y0'] - text_size, TEXT_COLOR,
                             text_size)

        output_img_path = './outputs/' + class_id + '_' + filename
        cv2.imwrite(output_img_path, img_c)

    streamManagerApi.DestroyAllStreams()
