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
import json
import argparse
from tqdm import trange

TEST_NUM = 19761

parser = argparse.ArgumentParser()
parser.add_argument("--RESULT_PATH", type=str)
parser.add_argument("--LABEL_PATH", type=str)
args = parser.parse_args()


file_set = set()
for file in os.listdir(args.RESULT_PATH):
    file_set.add(file)

video_label_dict = {}
with open(args.LABEL_PATH, "r") as fp:
    data = fp.read().split("\n")
    for d in data:
        if d == "":
            continue
        idx, label = d.split()
        video_label_dict[int(idx)] = int(label)

top1_count = 0
top5_count = 0
total_count = 0

UNWORK_LIST = set()
ERROR_count_dict = {}

for v in trange(TEST_NUM):
    try:
        total_count += 1
        score_sum_dict = {}
        for j in range(10):
            with open(f"{args.RESULT_PATH}/{v}_{j}.json", "r") as fp:
                res = json.load(fp)
            res = json.loads(res)
            for c in range(3):
                for k in range(5):
                    predict_idx = res["MxpiClass"][c*5+k]["classId"]
                    predict_score = res["MxpiClass"][c*5+k]["confidence"]
                    if predict_idx not in score_sum_dict:
                        score_sum_dict[predict_idx] = predict_score
                    else:
                        score_sum_dict[predict_idx] += predict_score
        score_sum_list = sorted(score_sum_dict.items(),
                                key=lambda t: t[1], reverse=True)

        for i in range(5):
            if score_sum_list[i][0] == video_label_dict[v]:
                top5_count += 1
                if i == 0:
                    top1_count += 1
                break
    except FileNotFoundError:
        UNWORK_LIST.add(v)
print("Top1:", top1_count/total_count)
print("Top5:", top5_count/total_count)
