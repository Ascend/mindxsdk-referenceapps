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
import multiprocessing
import subprocess
import time
import stat

parser = argparse.ArgumentParser()
parser.add_argument("--RESULT_SAVE_PATH", type=str)
parser.add_argument("--LOG_SAVE_PATH", type=str)
parser.add_argument("--FRAME_LENGTH_PATH", type=str)
parser.add_argument("--PROCESS_NUM", type=int)
parser.add_argument("--DEVICE_NUM", type=int)
args = parser.parse_args()

START_IDX = 0
END_IDX = 19761
TEST_NUM = END_IDX-START_IDX


if not os.path.exists(args.RESULT_SAVE_PATH):
    os.makedirs(args.RESULT_SAVE_PATH)

if not os.path.exists(args.LOG_SAVE_PATH):
    os.makedirs(args.LOG_SAVE_PATH)

frame_length_dict = {}
with open(args.FRAME_LENGTH_PATH, "r") as fp:
    data = fp.read().split("\n")
    for d in data:
        if d == "":
            continue
        _idx, _frame = d.split()
        frame_length_dict[int(_idx)] = int(_frame)

start_time = time.time()


def test_func(process_id, index_list, cross_process_num, cross_process_Lock):
    print(f"process {process_id} start")
    device_id = process_id % args.DEVICE_NUM
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    for idx in index_list:
        if idx >= END_IDX:
            break
        if idx not in frame_length_dict:
            continue
        frame = frame_length_dict[idx]
        remain = frame-13*5
        window_stride = 1
        if remain > 9:
            window_stride = int(remain/9)
        p = subprocess.Popen(["python3.9", "test_precision_sub.py", "--RESULT_SAVE_PATH", args.RESULT_SAVE_PATH, "--TEST_VIDEO_IDX", str(idx), "--DEVICE", str(device_id), "--WINDOW_STRIDE", str(window_stride)],
                             shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with os.fdopen(os.open(f"{args.LOG_SAVE_PATH}/{idx}.log", flags, modes), 'w') as fout:
            for line in p.stdout.readlines():
                fout.write(line.decode('UTF-8'))
        cross_process_Lock.acquire()
        cross_process_num.value += 1
        cost_time = int(time.time()-start_time)
        print(f"process_id: {process_id:<5} device_id: {device_id:<4} idx: {idx:<8} num:{cross_process_num.value:>5}/{TEST_NUM:<5} cost_time:{cost_time//3600:>2}h{(cost_time%3600)//60:>2}m{cost_time%60:>2}s")
        cross_process_Lock.release()


idx_list = [i for i in range(START_IDX, END_IDX)]
block_size = (TEST_NUM+args.PROCESS_NUM)//args.PROCESS_NUM
idx_block_list = [idx_list[i:i+block_size]
                  for i in range(0, TEST_NUM, block_size) if idx_list[i:i+block_size] != []]
processed_num = multiprocessing.Value('i', 0)
processed_lock = multiprocessing.Lock()
process_pool = [multiprocessing.Process(target=test_func, args=(
    idx_block, idx_block_list[idx_block], processed_num, processed_lock)) for idx_block in range(args.PROCESS_NUM)]

for process in process_pool:
    process.start()
for process in process_pool:
    process.join()
