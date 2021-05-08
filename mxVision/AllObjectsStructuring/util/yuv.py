#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2020 Huawei Technologies Co., Ltd
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

import cv2
import base64
import numpy as np

from PIL import Image


def yuv2bgr(image_b,
            height,
            width,
            clr_cvt_mtd="COLOR_YUV2BGR_NV12",
            output_path=None,
            base64_enc=False):
    yuv_vector = np.frombuffer(image_b, np.uint8)

    yuv_mtx = yuv_vector.reshape(height * 3 // 2, width)
    bgr_mtx = cv2.cvtColor(yuv_mtx, getattr(cv2, clr_cvt_mtd))

    if output_path:
        cv2.imwrite(output_path, bgr_mtx)

    if base64_enc:
        ret, buf = cv2.imencode(".jpg", bgr_mtx)
        img_bin = Image.fromarray(np.uint8(buf)).tobytes()
        encoded_ret = base64.b64encode(img_bin).decode()
        return encoded_ret

    if not (output_path or base64_enc):
        raise NotImplementedError("Please specify at least one method among "
                                  "saving and encoding image.")


if __name__ == '__main__':
    pass
