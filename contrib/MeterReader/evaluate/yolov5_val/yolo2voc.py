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

'''
把第一步得到的txt文件中的数据格式转换成voc格式
'''

import os
import stat
import numpy as np
import cv2

labelPath = '../evaluate/yolov5_val/det_val_data/det_sdk_txt'
imagePath = '../evaluate/yolov5_val/det_val_data/det_sdk_img'
vocPath = '../evaluate/yolov5_val/det_val_data/det_sdk_voc/'


# 坐标转换，原始存储的是YOLOv5格式
# Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
def xywhn2xyxy(boxes, width=800, height=800):
    padw = 0
    padh = 0
    y = np.copy(boxes)
    y[:, 0] = width * (boxes[:, 0] - boxes[:, 2] / 2) + padw  # top left x
    y[:, 1] = height * (boxes[:, 1] - boxes[:, 3] / 2) + padh  # top left y
    y[:, 2] = width * (boxes[:, 0] + boxes[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = height * (boxes[:, 1] + boxes[:, 3] / 2) + padh  # bottom right y
    return y


if __name__ == '__main__':
    folderlist = os.listdir(labelPath)
    for i in folderlist:
        label_path_new = os.path.join(labelPath, i)
        with open(label_path_new, 'r') as f:
            lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # predict_label
            print(lb)
        read_label = label_path_new.replace(".txt", ".jpg")
        read_label_path = read_label.replace('det_sdk_txt', 'det_val_img').replace("\\", "/")
        img = cv2.imread(read_label_path)
        h, w = img.shape[:2]
        lb[:, 1:] = xywhn2xyxy(lb[:, 1:], w, h)

        # 绘图
        for _, x in enumerate(lb):
            class_label = int(x[0])  # class
            cv2.rectangle(img, (round(x[1]), round(x[2])), (round(x[3]), round(x[4])), (0, 255, 0))
            cv2.putText(img, str(class_label),
                        (int(x[1]), int(x[2] - 2)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 0, 255),
                        thickness=2)
            with os.fdopen(os.open(vocPath + i, os.O_WRONLY | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR),
                           'a') as voc:
                voc.write(str(x[0]) + ' ' + str(x[5])
                          + ' ' + str(x[1]) + ' ' + str(x[2])
                          + ' ' + str(x[3]) + ' ' + str(x[4]) + '\n')
