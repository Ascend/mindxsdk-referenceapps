#!/usr/bin/env python
# coding=utf-8

"""
Copyright(C) Huawei Technologies Co.,Ltd. 2012-2022 All rights reserved.

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
import argparse
from tqdm import tqdm
from moviepy.editor import VideoFileClip

parser = argparse.ArgumentParser()
parser.add_argument("--video_path",type=str)
parser.add_argument("--save_path",type=str)
args = parser.parse_args()

file_list = []
for root, dirs, files in os.walk(args.video_path):
    for f in files:
        file_list.append(os.path.join(root, f))

with open(args.save_path, "w") as fp:
    i = 0
    error_list = []
    for f in tqdm(file_list):
        try:
            cap = VideoFileClip(f)
            # print(cap.duration)
            frame_num = int(cap.duration*25)
            fp.write(f'{i} {frame_num}\n')
        except:
            error_list.append(i)
        i += 1
    print(error_list) #note:data 3908 is broken.
