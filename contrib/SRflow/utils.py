#!/usr/bin/env python
# -*- encoding: utf-8 -*-
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

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr


def preprocess(path):
    image = cv2.imread(path)
    if image.shape[0] > 256 or image.shape[1] > 256:
        print("Error! Input image size > 256 * 256!")
        exit()
    image = cv2.copyMakeBorder(image, 0 , 256-image.shape[0] , 0 , 256-image.shape[1] , cv2.BORDER_REPLICATE)
    cv2.imwrite("tmp.jpg" , image)
    image = np.array(image).astype(np.float32).transpose()/255
    return image , image.shape


def postprocess(output , hr_size):
    res = output.reshape(3, 2048, 2048)
    y = (np.clip(res, 0, 1) * 255).astype(np.uint8)
    result = y.transpose([2, 1, 0])[
        :hr_size[0], :hr_size[1], :]
    return result


def valid(y , hr):
    psnr_val = psnr(y, hr)
    print('PSNR: {:.2f}'.format(psnr_val))
    return psnr_val

