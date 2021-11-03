#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
MIT License

Copyright (c) 2020 AIZOOTech

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Copyright(C) Huawei Technologies Co.,Ltd. 2012-2021 All rights reserved.

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

import numpy as np


def generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios, offset=0.5):
    """
    generate anchors.
    :param feature_map_sizes: list of list, for example: [[40,40], [20,20]]
    :param anchor_sizes: list of list, for example: [[0.05, 0.075], [0.1, 0.15]]
    :param anchor_ratios: list of list, for example: [[1, 0.5], [1, 0.5]]
    :param offset: default to 0.5
    :return:
    """
    anchor_bboxes = []
    for idx, feature_size in enumerate(feature_map_sizes):
        cx = (
            np.linspace(0, feature_size[0] - 1, feature_size[0]) + 0.5
        ) / feature_size[0]
        cy = (
            np.linspace(0, feature_size[1] - 1, feature_size[1]) + 0.5
        ) / feature_size[1]
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid_expend = np.expand_dims(cx_grid, axis=-1)
        cy_grid_expend = np.expand_dims(cy_grid, axis=-1)
        center = np.concatenate((cx_grid_expend, cy_grid_expend), axis=-1)

        num_anchors = len(anchor_sizes[idx]) + len(anchor_ratios[idx]) - 1
        center_tiled = np.tile(center, (1, 1, 2 * num_anchors))
        anchor_width_heights = []

        # different scales with the first aspect ratio
        for scale in anchor_sizes[idx]:
            ratio = anchor_ratios[idx][0]  # select the first ratio
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchor_width_heights.extend(
                [-width / 2.0, -height / 2.0, width / 2.0, height / 2.0]
            )

        # the first scale, with different aspect ratios (except the first one)
        for ratio in anchor_ratios[idx][1:]:
            s1 = anchor_sizes[idx][0]  # select the first scale
            width = s1 * np.sqrt(ratio)
            height = s1 / np.sqrt(ratio)
            anchor_width_heights.extend(
                [-width / 2.0, -height / 2.0, width / 2.0, height / 2.0]
            )

        bbox_coords = center_tiled + np.array(anchor_width_heights)
        bbox_coords_reshape = bbox_coords.reshape((-1, 4))
        anchor_bboxes.append(bbox_coords_reshape)
    anchor_bboxes = np.concatenate(anchor_bboxes, axis=0)
    return anchor_bboxes


if __name__ == "__main__":
    anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
