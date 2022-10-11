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
import stat
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source_path", type=str)
parser.add_argument("--target_path", type=str)
parser.add_argument("--label_path", type=str, default="K400_label.txt")
parser.add_argument("--save_path", type=str, default="video2label.txt")
args = parser.parse_args()


file_list = []
for root, dirs, files in os.walk(args.source_path):
    for f in files:
        file_list.append(os.path.join(root, f))

with open(args.label_path, "r") as fp:
    K400_label_str = fp.read()
    K400_label_list = K400_label_str.split("\n")
    assert len(K400_label_list) == 400
    K400_label_map = dict()
    for p, label in enumerate(K400_label_list):
        K400_label_map[label] = p

if not os.path.exists(args.target_path):
    os.makedirs(args.target_path)

flags = os.O_WRONLY | os.O_CREATE | os.O_EXCL
modes = stat.S_IWUSR | stat.S_IRUSR
with os.fdopen(os.open(args.save_path, flags, modes), 'w') as fout:
    i = 0
    for f in file_list:
        print(f)
        label = K400_label_map[f.split("/")[-2]]
        cmd = f'ffmpeg -i {f} -vcodec h264 -bf 0 -r 25 -an -f h264 {args.target_path}\\{i}.264 -y'
        result = os.popen(cmd).read().strip()
        fout.write(f'{i} {label}\n')
        i += 1
