# Copyright 2021 Huawei Technologies Co., Ltd
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

import os
import cv2

FILE_PATH = './BSD68/'
for file in os.listdir(FILE_PATH):
    img_path = FILE_PATH + file
    img = cv2.imread(img_path)
    # evaluate运行前需要执行下面的resize；main运行前注释掉下面一行代码
    img = cv2.resize(img, (480, 320))
    save_path = './dataset/' + file.split('.')[0] + '.jpg'
    cv2.imwrite(save_path, img)