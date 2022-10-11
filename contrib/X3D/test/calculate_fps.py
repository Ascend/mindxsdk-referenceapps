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
import json

parser = argparse.ArgumentParser()
parser.add_argument("--LOG_SAVE_PATH", type=str)
args = parser.parse_args()

SKIP_INIT_NUM = 1

total = []

for file in os.listdir(args.LOG_SAVE_PATH):
    if file.startswith("performance-statistics.log.plugin"):
        path = f"{args.LOG_SAVE_PATH}/{file}"
        with open(path, "r") as fp:
            throughput_data = []
            print(path)
            for json_line in fp.readlines():
                data = json.loads(json_line)
                if data["elementName"] == "mxpi_imagecrop0":
                    throughput_data.append(data)
            total += throughput_data

sum_freqyency = 0
sum_count = 0
for data in total:
    freq = int(data["frequency"])
    sum_freqyency += freq
    sum_count += 1

print(f"fps:{sum_freqyency/sum_count}")
