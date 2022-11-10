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
import argparse
import json
import os
import stat
from StreamManagerApi import StreamManagerApi

SAMPLE_NUM = 10

parser = argparse.ArgumentParser()
parser.add_argument("--RESULT_SAVE_PATH", type=str)
parser.add_argument("--TEST_VIDEO_IDX", type=int)
parser.add_argument("--DEVICE", type=int, default=0)
parser.add_argument("--WINDOW_STRIDE", type=int, default=1)
parser.add_argument("--RTSP_URL", type=str)
args = parser.parse_args()

streamManagerApi = StreamManagerApi()
ret = streamManagerApi.InitManager()
if ret != 0:
    print("Failed to init Stream manager, ret=%s" % str(ret))
    exit()

with open("../pipelines/testprecision.pipeline", 'rb') as f:
    pipeline_str = f.read()
pipeline = json.loads(pipeline_str)
pipeline["test_precision"]["stream_config"]["deviceId"] = str(args.DEVICE)
pipeline["test_precision"]["mxpi_rtspsrc0"]["props"]["rtspUrl"] = f"{args.RTSP_URL}{args.TEST_VIDEO_IDX}.264"
pipeline["test_precision"]["mxpi_x3dpreprocess0"]["props"]["windowStride"] = str(args.WINDOW_STRIDE)
pipeline_str = json.dumps(pipeline).encode()
ret = streamManagerApi.CreateMultipleStreams(pipeline_str)
if ret != 0:
    print("Failed to create Stream, ret=%s" % str(ret))
    exit()

STREAM_NAME = b'test_precision'
FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL
MODES = stat.S_IWUSR | stat.S_IRUSR
for i in range(SAMPLE_NUM):
    # Obtain the inference result by specifying STREAM_NAME and uniqueId.
    inferResult = streamManagerApi.GetResult(STREAM_NAME, 0, 1000000)
    if inferResult is None:
        break
    if inferResult.errorCode != 0:
        print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
            inferResult.errorCode, inferResult.data.decode()))
        break
    retStr = inferResult.data.decode()
    with os.fdopen(os.open(f"{args.RESULT_SAVE_PATH}//{args.TEST_VIDEO_IDX}_{i}.json", FLAGS, MODES), 'w') as fout:
        retJson = json.dump(retStr, fout)
    print(i, retStr)
streamManagerApi.DestroyAllStreams()
