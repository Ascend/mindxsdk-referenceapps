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
import stat
import subprocess
import json

parser = argparse.ArgumentParser()
parser.add_argument("--VIDEO_LIST_PATH", type=str, default="")
parser.add_argument("--LOG_SAVE_PATH", type=str, default="fps_test_log")
parser.add_argument("--TYPE", type=str, default="main")
parser.add_argument("--URL", type=str, default="")
parser.add_argument("--MAX_COUNT_IDX", type=int, default=50)
args = parser.parse_args()


def main():
    with open(args.VIDEO_LIST_PATH, "r") as fp:
        url_list = fp.read().strip().split()
    if not os.path.exists(args.LOG_SAVE_PATH):
        os.makedirs(args.LOG_SAVE_PATH)
    print("fps test start!")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    for idx, url in enumerate(url_list):
        p = subprocess.Popen(['python3.9', 'test_fps.py', '--TYPE', 'sub', '--URL', url, '--MAX_COUNT_IDX',
                             str(args.MAX_COUNT_IDX)], shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with os.fdopen(os.open(f"{args.LOG_SAVE_PATH}/{idx}.log", flags, modes), 'w') as fout:
            for line in p.stdout.readlines():
                fout.write(line.decode('UTF-8'))
        print(f"idx: {idx}, url: {url} test done!")


def sub():
    from StreamManagerApi import StreamManagerApi
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    with open("../pipelines/testperformance.pipeline", 'rb') as f:
        pipeline_str = f.read()
    pipeline = json.loads(pipeline_str)
    pipeline["test_performance"]["mxpi_rtspsrc0"]["props"]["rtspUrl"] = args.URL
    pipeline_str = json.dumps(pipeline).encode()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    stream_name = b'test_performance'
    idx = 0
    while idx < args.MAX_COUNT_IDX:
        infer_result = stream_manager_api.GetResult(stream_name, 0, 1000000)
        if infer_result is None:
            break
        if infer_result.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))
            break
        ret_str = infer_result.data.decode()
        idx += 1
        print(ret_str)
    stream_manager_api.DestroyAllStreams()


if __name__ == "__main__":
    if args.TYPE == "main":
        main()
    else:
        sub()
