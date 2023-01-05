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


from preprocess_utils import build_processor


img_scale = (768, 768)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=img_scale),
    dict(type='HWCToCHW', keys=['img']),
    dict(type='Collect', keys=['img']),
]

preprocessor = build_processor(test_pipeline)


def load_img_data(img_name_path, img_prefix_path=None):

    img_info = {'filename':img_name_path}
    img_data = {'img_prefix':img_prefix_path, 'img_info': img_info}

    resized_img_data = preprocessor(img_data)
    resize_img = resized_img_data.get('img', '')
    img_metas = resized_img_data.get('img_metas', '')
    return resize_img, img_metas
