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
parser.add_argument("--MUL_FACTOR", type=int, default=6)
args = parser.parse_args()

total = []

for file in os.listdir(args.LOG_SAVE_PATH):
    if file.startswith("performance-statistics.log.e2e"):
        path = f"{args.LOG_SAVE_PATH}/{file}"
        with open(path, "r") as fp:
            frequency = []
            for json_line in fp.readlines():
                data = json.loads(json_line)
                freq = int(data["frequency"])
                if freq > 1:  # 0 or 1 is unstable state.
                    frequency.append(freq)
            total += frequency[2:-1]  # remove start or end data.
print(f"fps:{sum(total)*args.MUL_FACTOR/len(total)}")
