# -*- coding: utf-8 -*-
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

import os
import numpy as np
from utils import polyiou

def getfilefromthisrootdir(directory, ext=None): 
    allfiles = []
    needExtFilter = (ext != None)
    for root, dirs, files in os.walk(directory):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles

def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])

def py_cpu_nms_poly(dets, thresh):
    """
    任意四点poly nms.取出nms后的边框的索引
    @param dets: shape(detection_num, [poly, confidence1]) 原始图像中的检测出的目标数量
    @param thresh:
    @return:
            keep: 经nms后的目标边框的索引
    """  
    scores = dets[:, 8]
    polys = []

    for det in dets:
        tm_polygon = [det[0], det[1],
                      det[2], det[3],
                      det[4], det[5],
                      det[6], det[7]]
        polys.append(tm_polygon)
    # argsort将元素小到大排列 返回索引值 [::-1]即从后向前取元素
    order = scores.argsort()[::-1]  # 取出元素的索引值 顺序为从大到小
    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]  # 取出当前剩余置信度最大的目标边框的索引
        keep.append(i)
        for j in range(order.size - 1):  # 求出置信度最大poly与其他所有poly的IoU
            iou = polyiou.iou_poly(polys[i], polys[order[j + 1]])
            ovr.append(iou)
        ovr = np.array(ovr)
        inds = np.where(ovr <= thresh)[0]  # 找出iou小于阈值的索引
        order = order[inds + 1]
    return keep

def poly2origpoly(poly, x, y, rate):
    origpoly = []
    for i in range(int(len(poly)/2)):
        tmp_x = float(poly[i * 2] + x) / float(rate)
        tmp_y = float(poly[i * 2 + 1] + y) / float(rate)
        origpoly.append(tmp_x)
        origpoly.append(tmp_y)
    return origpoly

def nmsbynamedict(nameboxdict, nameboxdict_classname, nms, thresh):
    """
    对namedict中的目标信息进行nms.不改变输入的数据形式
    @param nameboxdict: eg:{
                           'P706':[[poly1, confidence1], ..., [poly9, confidence9]],
                           ...
                           'P700':[[poly1, confidence1], ..., [poly9, confidence9]]
                            }
    @param nameboxdict_classname: eg:{
                           'P706':[[poly1, confidence1,'classname'], ..., [poly9, confidence9, 'classname']],
                           ...
                           'P700':[[poly1, confidence1, 'classname'], ..., [poly9, confidence9, 'classname']]
                            }
    @param nms:
    @param thresh: nms阈值, IoU阈值
    @return:
            nameboxnmsdict: eg:{
                                'P706':[[poly1, confidence1, 'classname'], ..., [poly_nms, confidence9, 'classname']],
                                 ...
                                'P700':[[poly1, confidence1, 'classname'], ..., [poly_nms, confidence9, 'classname']]
                               }
    """
    # 初始化字典
    nameboxnmsdict = {x: [] for x in nameboxdict}  # eg: nameboxnmsdict={'P0770': [], 'P1888': []}
    for imgname in nameboxdict:  # 提取nameboxdict中的key eg:P0770   P1888
        keep = nms(np.array(nameboxdict[imgname]), thresh)  # rotated_nms索引值列表
        outdets = []
        for index in keep:
            outdets.append(nameboxdict_classname[imgname][index])
        nameboxnmsdict[imgname] = outdets
    return nameboxnmsdict