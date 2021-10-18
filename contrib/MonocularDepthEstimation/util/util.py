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


def bilinear_sampling(source, width_extend_multiple=2, height_extend_multiple=2):
    """
    Bilinear sampling of source data

        |   upper left           |
     y2 |-------P1-----|---------P2---
        |       |      |         |
        |       |      |         |
      y |--------------P-------------
        |       |      |         |
        |       |      |         |
     y1 |-------P3---------------P4---
        |       |      |         | lower right
        |       |      |         |
        |----------------------------
               x1      x        x2
    f(x,y) = (1 - w1) * (1 - w2) * value(P1) +
             (1 - w1) * w2 * value(P2) +
             w1 * (1 - w2) * value(P3) +
             w1 * w2 * value(P4)

    :param source: source data
    :param width_extend_multiple: A multiple of the width expansion relative to the source data
    :param height_extend_multiple: A multiple of the height expansion relative to the source data
    :return: output data after bilinear sampling
    """
    # source data size
    src_capacity = source.shape[0]
    src_height = source.shape[1]
    src_width = source.shape[2]
    # destination data size
    dst_height = src_height * height_extend_multiple
    dst_width = src_width * width_extend_multiple

    # scale factor
    scale_height = src_height / dst_height
    scale_width = src_width / dst_width

    # calculate the corresponding coordinate of source data
    x_index = np.array([x for x in range(dst_width)])
    y_index = np.array([y for y in range(dst_height)])
    src_x = (x_index + 0.5) * scale_width - 0.5
    src_y = (y_index + 0.5) * scale_height - 0.5
    src_x = np.repeat(np.expand_dims(src_x, axis=0), dst_height, axis=0)
    src_y = np.repeat(np.expand_dims(src_y, axis=1), dst_width, axis=1)

    # rounded down, get the row and column number of upper left corner
    src_x_int = np.floor(src_x)
    src_y_int = np.floor(src_y)
    # take out the decimal part to construct the weight
    src_x_float = src_x - src_x_int
    src_y_float = src_y - src_y_int
    # expand to input data size
    src_x_float = np.repeat(np.expand_dims(src_x_float, axis=0), src_capacity, axis=0)
    src_y_float = np.repeat(np.expand_dims(src_y_float, axis=0), src_capacity, axis=0)

    # get upper left and lower right index
    left_x_index = src_x_int.astype(int)
    upper_y_index = src_y_int.astype(int)
    right_x_index = left_x_index + 1
    lower_y_index = upper_y_index + 1

    # boundary condition
    left_x_index[left_x_index < 0] = 0
    upper_y_index[upper_y_index < 0] = 0
    right_x_index[right_x_index > src_width - 1] = src_width - 1
    lower_y_index[lower_y_index > src_height - 1] = src_height - 1

    # upper left corner data
    upper_left_value = source[:, upper_y_index, left_x_index]
    # upper right corner data
    upper_right_value = source[:, upper_y_index, right_x_index]
    # lower left corner data
    lower_left_value = source[:, lower_y_index, left_x_index]
    # lower right corner data
    lower_right_value = source[:, lower_y_index, right_x_index]

    # bilinear sample
    target = (1. - src_y_float) * (1. - src_x_float) * upper_left_value + \
             (1. - src_y_float) * src_x_float * upper_right_value + \
             src_y_float * (1. - src_x_float) * lower_left_value + \
             src_y_float * src_x_float * lower_right_value

    return target.squeeze()


def colorize(value, value_min=10, value_max=1000, color_depth=255, extend_to_bgr=False, dominant_color=100):
    """
    normalize source value and assign color values according to the source data value and config
    :param value: source value
    :param value_min: min value of source value
    :param value_max: max value of source value
    :param color_depth: color depth need to use
    :param extend_to_bgr: whether extend to bgr channels
    :param dominant_color: color channels need to use, each bit number represents
            whether use corresponding color channel (order is blue, green and red)
            for example: (1) 100, which represents that use blue, but not use green and red
                         (2) 101, which represents that use blue and red, but not use green
    :return: output value after colorizing
    """
    # normalize
    value_min = value.min() if value_min is None else value_min
    value_max = value.max() if value_max is None else value_max
    if value_min != value_max:
        value = (value - value_min) / (value_max - value_min)
    else:
        # Avoid 0-division
        value = value * 0.

    value = np.clip((value * color_depth), 0, color_depth)

    # bgr color mask code
    blue_mask = 100
    green_mask = 10
    red_mask = 1
    if extend_to_bgr:
        use_blue = int(dominant_color / blue_mask) == 1
        dominant_color = dominant_color % blue_mask
        use_green = int(dominant_color / green_mask) == 1
        dominant_color = dominant_color % green_mask
        use_red = int(dominant_color / red_mask) == 1

        img = np.zeros((value.shape[0], value.shape[1], 3))
        if use_blue:
            img[:, :, 0] = value
        if use_green:
            img[:, :, 1] = value
        if use_red:
            img[:, :, 2] = value

        return img
    else:
        return value
