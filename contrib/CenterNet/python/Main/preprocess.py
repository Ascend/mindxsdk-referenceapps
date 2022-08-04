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


def preproc(image):
    height, width = image.shape[0: 2]
    new_height = int(height)
    new_width  = int(width)
    inp_height, inp_width = 512, 512
    c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.array([[256, 256], [256, 0], [0, 0]])
    src[0, :] = c
    src[1, :] = np.array([c[0], c[1] - s * 0.5])
    src[2, :] = c - np.array([s * 0.5, s * 0.5])
    trans_input = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
	resized_image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - [[[0.40789655, 0.44719303, 0.47026116]]]) /
                                     [[[0.2886383, 0.27408165, 0.27809834]]]).astype(np.float32)
    
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    img = np.array(images).astype(np.float32)
    return img



