#!/usr/bin/env python
# coding=utf-8

"""
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
"""

import cv2
import numpy as np
import time
from enum import Enum
import copy
import math


class Cvcolors(Enum):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    PURPLE = (247, 44, 200)
    ORANGE = (44, 162, 247)
    MINT = (239, 255, 66)
    YELLOW = (2, 255, 250)
    DINGXIANG = (204, 164, 227)


KITTI_COLOR_MAP = (
    Cvcolors.RED.value,
    Cvcolors.GREEN.value,
    Cvcolors.BLUE.value,
    Cvcolors.PURPLE.value,
    Cvcolors.ORANGE.value,
    Cvcolors.MINT.value,
    Cvcolors.YELLOW.value,
    Cvcolors.DINGXIANG.value
)


def cv_draw_bbox3d_rtm3d(img, m_cls, m_scores, v_projs, label_map=None, color_map=KITTI_COLOR_MAP):
    """
    将传入的目标信息逐个的传给下一个可视化函数
    img：输入图片， m_cls:类别， m_scores:置信度， v_projs:框信息， label_map:类别标签, color_map：颜色对应的rgb值
    """
    for m_cls_i, m_scores_i, v_projs_i in zip(m_cls, m_scores, v_projs):
        label = label_map[m_cls_i] if label_map is not None else m_cls_i
        cv_draw_bbox_3d_kitti(img, label, v_projs_i.astype(np.int),
                              center=None, color=color_map[m_cls_i],
                              thickness=2)


def cv_draw_bbox_3d_kitti(img, cls, proj_2d, center=None, color=(0, 0, 255), thickness=1):
    """
    对传入的目标进行画框标注
    img：输入图片, cls：检测物体实际名称, proj_2d：每个目标的框信息, 默认center=None, color：颜色对应的rgb值, thickness：层数
    """
    tl = thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    # to see the corners on image as red circles

    outline = [0, 1, 3, 2, 0, 4, 5, 7, 6, 4, 5, 1, 3, 7, 6, 2]
    outline_pt = proj_2d[outline]
    for i in range(len(outline) - 1):
        cv2.line(img, (outline_pt[i][0], outline_pt[i][1]), (outline_pt[i + 1][0], outline_pt[i + 1][1]), color, tl)

    front_mark = np.array([[proj_2d[0][0], proj_2d[0][1]],
                           [proj_2d[1][0], proj_2d[1][1]],
                           [proj_2d[3][0], proj_2d[3][1]],
                           [proj_2d[2][0], proj_2d[2][1]]
                           ], dtype=np.int)
    front_mark = [front_mark]

    mask = np.copy(img)
    cv2.drawContours(mask, front_mark, -1, color, thickness=cv2.FILLED, lineType=cv2.LINE_8)
    rate = 0.7
    res = rate * img.astype(np.float) + (1 - rate) * mask.astype(np.float)
    np.copyto(img, res.astype(np.uint8))

    label = '{}:({:.2f},{:.2f},{:.2f})'.format(cls, center[0], center[1], center[2]) if center is not None else \
        '{}'.format(cls)
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    box_3d = np.array(proj_2d).min(axis=0)
    c1 = (box_3d[0], box_3d[1])
    c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)




