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

import cv2 as cv
import numpy as np
import os
from PIL import Image, ImageDraw

def img_version():
    image_path = './test.jpg'
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    path = "./test"
    lists = os.listdir(path)
    file_name = []
    left = 0
    right = 8
    for files in lists:
        if files.endswith('.txt'):
            file_name.append(files)
    for txt in file_name:
        f = open(path + '/' + txt, 'r')
        line = f.readline()
        while line:
            rect = []
            line = line.replace(' ', '').split(',')
            for i in range(0, len(line), 2):
                rect.append((int(line[i]), int(line[i + 1])))
            draw.polygon(rect, outline = (255, 0, 0))
            # 数字8用以控制txt文件的读取，每次读取到一个点对
            left += 8
            right += 8
            line = f.readline()
        f.close()

    image.save("./my_test.jpg")
    image.close()
