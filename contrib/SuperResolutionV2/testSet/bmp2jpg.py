#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreimport json
import os
import sys
from PIL import Image

if __name__ == '__main__':
    # test image set path
    TEST_IMAGE_SET_PATH = "Set14"
    # parse command arguments
    if len(sys.argv) == 2:
        if sys.argv[1] == '':
            print('test image set path is not valid, use default config.')
        else:
            TEST_IMAGE_SET_PATH = sys.argv[1]
    # jpg format image set path
    output_image_set_path = TEST_IMAGE_SET_PATH + "-jpg"
    # check output paths
    if os.path.exists(output_image_set_path) != 1:
        print('The image set path {} does not exist, create it.'.format(output_image_set_path))
        os.mkdir(output_image_set_path)

    # get all image files
    image_files = os.listdir(TEST_IMAGE_SET_PATH)
    # convert bmp to jpg
    for test_image_path in image_files:
        image_file = TEST_IMAGE_SET_PATH + "/" + test_image_path
        out_img = Image.open(image_file).convert("RGB")
        out_img.save(output_image_set_path + "/" + test_image_path[:-4] + ".jpg")