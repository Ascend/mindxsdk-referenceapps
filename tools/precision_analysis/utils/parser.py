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

import os
from json import loads


def crnn_parsing_func(ret, shared_params):
    try:
        ret = loads(ret)
        return ret["MxpiTextsInfo"][0]["text"][0]
    except BaseException:
        return ""


def ssd_mobilenet_fpn_parsing_func(ret, shared_params):
    """
    Basic format is as follow:
    [{“bbox”: [225.7, 207.6, 128.7, 140.2], “score”: 0.999, “image_id”: 581929, “category_id”: 17},
    {“bbox”: [231.1, 110.6, 33.5, 36.7], “score”: 0.992, “image_id”: 581929, “category_id”: 17},…]
    :param ret:
    :param shared_params:
    :return:
    """
    ret = loads(ret)
    objs = ret.get("MxpiObject")
    if not isinstance(objs, list):
        return []

    annotations = []
    for obj in objs:
        x0, y0 = obj.get("x0"), obj.get("y0")
        x1, y1 = obj.get("x1"), obj.get("y1")
        w, h = x1 - x0, y1 - y0
        bbox = [x0, y0, w, h]
        score = obj.get("classVec")[0].get("confidence")
        file_name = shared_params.get("file_name")
        image_id = int(os.path.splitext(file_name)[0])
        category_id = obj.get("classVec")[0].get("classId")
        ann = {
            "bbox": bbox,
            "score": score,
            "image_id": image_id,
            "category_id": category_id
        }

        annotations.append(ann)

    return annotations


def label_mapping_for_coco(k):
    if k >= 1 and k <= 11:
        class_id = k
    elif k >= 12 and k <= 24:
        class_id = k + 1
    elif k >= 25 and k <= 26:
        class_id = k + 2
    elif k >= 27 and k <= 40:
        class_id = k + 4
    elif k >= 41 and k <= 60:
        class_id = k + 5
    elif k == 61:
        class_id = k + 6
    elif k == 62:
        class_id = k + 8
    elif k >= 63 and k <= 73:
        class_id = k + 9
    elif k >= 74 and k <= 80:
        class_id = k + 10
    else:
        raise ValueError("Category id is out of memory.")

    return class_id


def ctpn_parse_and_save_func(ret, shared_params):
    """
    Basic format is as follow:
    {"MxpiTextObject":[{"confidence":0.98540580300000002,"text":"","x0":341.33334400000001,"x1":1109.3333700000001,
    "x2":1109.3333700000001,334400000001,"y0":766.46466099999998,"y1":766.46466099999998,"y2":795.03021200000001,
    "y3":795.03021200000001},{"confidence":0.965757309ext":"","x0":405.33331299999998,"x1":1066.6666299999999,
    "x2":1066.6666299999999,"x3":405.33331299999998,"y0":741.29064900000003,"y1":7400003,"y2":767.99176,
    "y3":767.99176}]}
    :param ret:
    :param shared_params:
    :return: None
    """
    ret = loads(ret)
    result_path = shared_params['result_path']
    file_name = shared_params['file_name'].strip().split('.')[0]
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    objs = ret.get("MxpiTextObject")
    if objs is None:
        with open(os.path.join(result_path, 'res_' + file_name + '.txt'), 'w') as f_write:
            f_write.write("")
        return True
    boxes = []
    for res in objs:
        boxes.append([int(res['x0']), int(res['y0']), int(res['x2']), int(res['y2'])])
    with open(os.path.join(result_path, 'res_' + file_name + '.txt'), 'w') as f_write:
        for i, box in enumerate(boxes):
            line = ",".join(str(box[k]) for k in range(4))
            f_write.writelines(line)
            f_write.write('\n')
    return True
