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

import torch
import torch.nn.functional as F


def bilinear_sampling(source, width_extend_multiple=2, height_extend_multiple=2):
    """
    Bilinear sampling of source data
    :param source: source data
    :param width_extend_multiple: A multiple of the width expansion relative to the source data
    :param height_extend_multiple: A multiple of the height expansion relative to the source data
    :return: output data after bilinear sampling
    """
    h = torch.linspace(-1, 1, source.shape[1] * height_extend_multiple)
    w = torch.linspace(-1, 1, source.shape[2] * width_extend_multiple)
    meshx, meshy = torch.meshgrid((h, w))
    grid = torch.stack((meshy, meshx), 2)
    grid = grid.unsqueeze(0)

    input_data = np.zeros((1, source.shape[0], source.shape[1], source.shape[2]))
    input_data[0] = source
    input_data = torch.from_numpy(input_data).float()
    output_data = F.grid_sample(input_data, grid, align_corners=True)

    return output_data.squeeze().numpy()


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
    if extend_to_bgr:
        use_blue = int(dominant_color / 100) == 1
        dominant_color = dominant_color % 100
        use_green = int(dominant_color / 10) == 1
        dominant_color = dominant_color % 10
        use_red = int(dominant_color / 1) == 1

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
