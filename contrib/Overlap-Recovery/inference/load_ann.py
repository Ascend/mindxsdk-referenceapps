# -*- coding: utf-8 -*-

# Copyright 2022 Huawei Technologies Co., Ltd
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


import json
import os.path as osp
import imagesize


def load_annotations(ann_file_path, img_prefix_path, seg_prefix_path):
    """Load annotation from Overlap"""
    data_result_list = []
    img_dir = img_prefix_path
    seg_dir = seg_prefix_path
    if osp.isfile(ann_file_path):
        with open(ann_file_path, 'r', encoding='utf-8') as f:
            info_list = json.load(f)
        for info_ in info_list:
            assert len(info_) == 3, f"Invalid line: {info_}"
            img_name = info_['img_name']
            data_info = dict(img_path=osp.join(img_dir, img_name))
            data_info['data_type'] = info_['data_type']
            data_info['filename'] = img_name
            width, height = imagesize.get(data_info.get('img_path', ''))
            data_info['width'] = width
            data_info['height'] = height
            seg_map_path = []
            text_labels = []
            bboxes = []
            # should follow a pre-defined order, e.g., from top layer to bottom
            for text_ins in info_['texts']:
                x, y, w, h = text_ins['bbox']
                bbox = [x, y, x + w, y + h]
                bboxes.append(bbox)
                seg_map_path.append(osp.join(seg_dir, text_ins[f"mask"]))
                text_labels.append(text_ins['label'])
            data_info['bboxes'] = bboxes
            data_info['seg_map_path'] = seg_map_path
            data_info['text_labels'] = text_labels
            data_result_list.append(data_info)
    else:
        raise NotImplementedError
    return data_result_list
