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

import h5py
import imageio
import os
import numpy as np

if __name__ == '__main__':

    data_file_path = '../test_set/nyu_depth_v2_labeled.mat'
    output_image_path = '../test_set/image'
    output_depth_info_path = '../test_set/depth_info'

    # check origin data file
    if os.path.exists(data_file_path) != 1:
        error_message = 'The {} does not exist'.format(data_file_path)
        raise FileNotFoundError(error_message)

    # load data file
    data = h5py.File(data_file_path)

    # get image data
    image = np.array(data['images'])
    # get depth info data
    depth = np.array(data['depths'])

    print(image.shape)
    image = np.transpose(image, (0, 2, 3, 1))
    print(image.shape)

    print(depth.shape)

    # save image
    for i in range(image.shape[0]):
        index = str(i)
        image_index_path = output_image_path + index + '.jpg'
        out_img = image[i, :, :, :]
        out_img = out_img.transpose(1, 0, 2)
        imageio.imwrite(image_index_path, out_img)

    # save depth info
    for i in range(depth.shape[0]):
        index = str(i)
        depth_index_path = output_depth_info_path + index + '.npy'
        out_depth = depth[i, :, :]
        out_depth = out_depth.transpose(1, 0)
        np.save(depth_index_path, out_depth)
