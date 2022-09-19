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
    output_path = '../pic/MuPoTS-3D.json'
    img_path = '../pic/MultiPersonTestSet'
    with open(output_path, 'r') as f:
        output = json.load(f)
    image = output['images']
    ann = output['annotations']

    m =0
    len_image = len(image)
    for n in range(0, len_image):
        conf = configparser.ConfigParser()
        image_intrinsic = image[n]['intrinsic']
        conf.add_section('intrinsic')  # 添加section
        conf.set('intrinsic', 'intrinsic', str(image_intrinsic))

        image_id = image[n]['id']
        count = 1
        conf.add_section('keypoints_cam')  # 添加section
        conf.add_section('keypoints_img')  # 添加section
        len_ann = len(ann)
        for m in range(0, len_ann):
            if ann[m]['image_id'] == image_id:
                keypoints_cam = ann[m]['keypoints_cam']
                conf.set('keypoints_cam', 'keypoints_cam'+str(count), str(keypoints_cam))
                keypoints_img = ann[m]['keypoints_img']
                conf.set('keypoints_img', 'keypoints_img'+str(count), str(keypoints_img))
                count += 1
            m += 1
        conf.add_section('image_id')  # 添加section
        conf.set('image_id', 'image_id', str(image_id))
        file_name = image[n]['file_name']
        save_path = img_path + '/' + file_name[:-4] + '.ini'
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(save_path, flags, modes), 'w', encoding='utf-8') as f:
            conf.write(f)
        print('have made' + str(n))
