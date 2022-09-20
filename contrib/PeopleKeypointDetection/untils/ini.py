#!/usr/bin/env python
# coding=utf-8

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
import stat
import json
import configparser

if __name__ == '__main__':
    OUTPUT_PATH = '../pic/MuPoTS-3D.json'
    IMG_PATH = '../pic/MultiPersonTestSet'
    with open(OUTPUT_PATH, 'r') as f:
        output = json.load(f)
    image = output['images']
    ann = output['annotations']

    MM_NUM = 0
    len_image = len(image)
    for n in range(0, len_image):
        conf = configparser.ConfigParser()
        image_intrinsic = image[n]['intrinsic']
        conf.add_section('intrinsic')  # 添加section
        conf.set('intrinsic', 'intrinsic', str(image_intrinsic))

        image_id = image[n]['id']
        COUNT_NU = 1
        conf.add_section('keypoints_cam')  # 添加section
        conf.add_section('keypoints_img')  # 添加section
        len_ann = len(ann)
        for MM_NUM in range(0, len_ann):
            if ann[MM_NUM]['image_id'] == image_id:
                keypoints_cam = ann[MM_NUMm]['keypoints_cam']
                conf.set('keypoints_cam', 'keypoints_cam'+str(COUNT_NU), str(keypoints_cam))
                keypoints_img = ann[MM_NUM]['keypoints_img']
                conf.set('keypoints_img', 'keypoints_img'+str(COUNT_NU), str(keypoints_img))
                COUNT_NU += 1
            MM_NUM += 1
        conf.add_section('image_id')  # 添加section
        conf.set('image_id', 'image_id', str(image_id))
        file_name = image[n]['file_name']
        save_path = IMG_PATH + '/' + file_name[:-4] + '.ini'
        FLAG_NUM = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        MODE_NUM = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(save_path, flFLAG_NUM, MODE_NUM), 'w', encoding='utf-8') as f:
            conf.write(f)
        print('have made' + str(n))
