# -*- coding: utf-8 -*-
#!/usr/bin/env python
# coding=utf-8

"""
 Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.

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
from PIL import Image


filePath = 'image/images'
jpg_name = os.listdir(filePath)

if __name__ == '__main__':
    # Convert PNG format pictures to JPG format pictures
    for name in jpg_name:
        im = Image.open('image/images/' + name)
        im = im.convert('RGB')
        im.save('image_jpg/' + name[0:5] + '.jpg', quality = 95)
        print(name[0:5])
