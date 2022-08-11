#!/usr/bin/env python
# coding=utf-8


# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import os
import cv2
import numpy as np


def preproc(my_i):
    my_h, my_w = my_i.shape[0: 2]
    my_new_h = int(my_h)
    my_new_w  = int(my_w)
    #目标尺寸 512 512
    my_inp_h, my_inp_w = 512, 512
    c = np.array([my_new_w / 2., my_new_h / 2.], dtype=np.float32)
    s = max(my_h, my_w) * 1.0
    my_s = np.zeros((3, 2), dtype=np.float32)
    my_d = np.array([[256, 256], [256, 0], [0, 0]])
    #余弦仿射
    my_s[0, :] = c
    my_s[1, :] = np.array([c[0], c[1] - s * 0.5])
    my_s[2, :] = c - np.array([s * 0.5, s * 0.5])
    trans_input = cv2.getAffineTransform(np.float32(my_s), np.float32(my_d))
    #调整回原始尺寸
    resized_image = cv2.resize(my_i, (my_new_w, my_new_h))
    inp_image = cv2.warpAffine(
	resized_image, trans_input, (my_inp_w, my_inp_h), flags=cv2.INTER_LINEAR)
    #标准化
    inp_image = ((inp_image / 255. - [[[0.40789655, 0.44719303, 0.47026116]]]) /
                                     [[[0.2886383, 0.27408165, 0.27809834]]]).astype(np.float32)
    
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, my_inp_h, my_inp_w)
    img = np.array(images).astype(np.float32)
    return img



