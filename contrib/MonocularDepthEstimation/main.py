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

import io
import os
import sys

import cv2
from PIL import Image

from depth_estimation.monocular_depth_estimation import depth_estimation
from util.util import bilinear_sampling
from util.util import colorize

if __name__ == '__main__':

    input_image_path = 'image/${测试图片文件名}'
    output_result_path = "result/${输出结果文件名}"

    # parse command arguments
    if len(sys.argv) == 2:
        if sys.argv[1] == '':
            print('input image path is valid, use default config.')
        else:
            input_image_path = sys.argv[1]
        print('no output result path, use default config.')
    elif len(sys.argv) == 3:
        if sys.argv[1] == '':
            print('input image path is valid, use default config.')
        else:
            input_image_path = sys.argv[1]
        if sys.argv[2] == '':
            print('output result path is valid, use default config.')
        else:
            output_result_path = sys.argv[2]
    else:
        print("Please enter at least image path, "
              "such as 'python3.7 main.py image/test.jpg' or 'bash run.sh -m infer -i image/test.jpg'.")
        print('no input image path and output result path, use default config.')

    print('input image path: {}.'.format(input_image_path))
    print('output result path: {}.'.format(output_result_path))

    # check input image
    if os.path.exists(input_image_path) != 1:
        error_message = 'The {} does not exist'.format(input_image_path)
        raise FileNotFoundError(error_message)
    else:
        try:
            image = Image.open(input_image_path)
            if image.format != 'JPEG':
                raise AssertionError('input image only support jpg')
            elif image.width < 32 or image.width > 8192:
                raise AssertionError('input image width must in range [32, 8192], curr is {}'.format(image.width))
            elif image.height < 32 or image.height > 8192:
                raise AssertionError('input image height must in range [32, 8192], curr is {}'.format(image.height))
            else:
                # read input image bytes
                image_bytes = io.BytesIO()
                image.save(image_bytes, format='JPEG')
                input_image_data = image_bytes.getvalue()
        except IOError:
            raise IOError(
                'an IOError occurred while opening {}, maybe your input is not a picture'.format(input_image_path))

    # depth estimation
    depth_pic_array, input_image_info = depth_estimation(input_image_data)

    # get size of input image and output depth image
    input_image_height = input_image_info[0][0]
    input_image_width = input_image_info[0][1]

    print('save infer depth picture start.')

    # double linear sample to reconstruct depth pic whose size is approximate to input size
    depth_pic_array = bilinear_sampling(depth_pic_array, input_image_width, input_image_height)

    # colorize by inferred depth result
    is_extend_to_bgr = True
    use_dominant_color = 111
    output_depth_pic = colorize(depth_pic_array, value_min=None, value_max=None, extend_to_bgr=is_extend_to_bgr,
                                dominant_color=use_dominant_color)

    # save depth pic
    if is_extend_to_bgr:
        cv2.imwrite(output_result_path, output_depth_pic)
    else:
        depth_pic = Image.fromarray(output_depth_pic)
        if depth_pic.mode == 'F':
            depth_pic = depth_pic.convert('RGB')
        depth_pic.save(output_result_path)

    print('save infer depth picture end.')
