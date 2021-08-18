# Copyright 2021 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import cv2

with open("ImageSets/Main/test.txt", "r") as f:
    data = f.readlines()
    text_data = []
    for line in data:
        line_new = line.strip('\n')  # Remove the newline character of each element in the list
        text_data.append(line_new)
    print(text_data)

path = 'JPEGImages'
save_path = 'TestImages'

for item in os.listdir(path):
    file_name = item.split('.')[0]
    if file_name in text_data:
        img = cv2.imread(path + '/' + item)
        cv2.imwrite(save_path + '/' + file_name + ".jpg", img)