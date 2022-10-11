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
flags = parser.parse_args()

streamManagerApi = StreamManagerApi()
ret = streamManagerApi.InitManager()
pipeline = {
    "prec_verify": {
        "stream_config": {
            "deviceId": f"{flags.DEVICE}"
        },
        "mxpi_rtspsrc0": {
            "factory": "mxpi_rtspsrc",
            "props": {
                "rtspUrl": f"rtsp://192.168.88.107:8554/{flags.TEST_VIDEO_IDX}.264",
                "channelId": "0",
                "timeout": "1"
            },
            "next": "mxpi_videodecoder0"
        },
        "mxpi_videodecoder0": {
            "factory": "mxpi_videodecoder",
            "props": {
                "inputVideoFormat": "H264",
                "outputImageFormat": "YUV420SP_NV21",
                "vdecChannelId": "0",
                "outMode": "1"
            },
            "former": "mxpi_rtspsrc0",
            "next": "mxpi_x3dpreprocess0"
        },
        "mxpi_x3dpreprocess0": {
            "props": {
                "dataSource": "mxpi_videodecoder0",
                "skipFrameNum": "5",
                "windowStride": f"{flags.WINDOW_STRIDE}"
            },
            "factory": "mxpi_x3dpreprocess",
            "next": "mxpi_tensorinfer0"
        },
        "mxpi_tensorinfer0": {
            "props": {
                "dataSource": "mxpi_x3dpreprocess0",
                "modelPath": "../models/x3d/x3d_s1_test.om",
                "singleBatchInfer": "1",
                "waitingTime": "250000"
            },
            "factory": "mxpi_tensorinfer",
            "next": "mxpi_classpostprocessor0"
        },
        "mxpi_classpostprocessor0": {
            "props": {
                "dataSource": "mxpi_tensorinfer0",
                "postProcessConfigPath": "../models/x3d/x3d_post_test.cfg",
                "labelPath": "../models/x3d/kinetics400.names",
                "postProcessLibPath": "../../../lib/modelpostprocessors/libx3dpostprocess.so"
            },
            "factory": "mxpi_classpostprocessor",
            "next": "mxpi_dataserialize0"
        },
        "mxpi_dataserialize0": {
            "props": {
                "outputDataKeys": "mxpi_classpostprocessor0"
            },
            "factory": "mxpi_dataserialize",
            "next": "appsink0"
        },
        "appsink0": {
            "props": {
                "blocksize": "4096000"
            },
            "factory": "appsink"
        }
    }
}
pipelineStr = json.dumps(pipeline).encode()
ret = streamManagerApi.CreateMultipleStreams(pipelineStr)

streamName = b'prec_verify'
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
modes = stat.S_IWUSR | stat.S_IRUSR
for i in range(SAMPLE_NUM):
    # Obtain the inference result by specifying streamName and uniqueId.
    inferResult = streamManagerApi.GetResult(streamName, 0, 1000000)
    if inferResult is None:
        break
    if inferResult.errorCode != 0:
        print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
            inferResult.errorCode, inferResult.data.decode()))
        break
    retStr = inferResult.data.decode()
    with os.fdopen(os.open(f"{flags.RESULT_SAVE_PATH}//{flags.TEST_VIDEO_IDX}_{i}.json", flags, modes), 'w') as fout:
        retJson = json.dump(retStr, fout)
    print(i, retStr)
streamManagerApi.DestroyAllStreams()
