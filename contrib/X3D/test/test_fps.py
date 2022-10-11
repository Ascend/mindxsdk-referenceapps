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
    flags = os.O_WRONLY | os.O_CREATE | os.O_EXCL
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
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    pipeline = {
        "detection+action recognition": {
            "stream_config": {
                "deviceId": "0"
            },
            "mxpi_rtspsrc0": {
                "props": {
                    "rtspUrl": args.URL
                },
                "factory": "mxpi_rtspsrc",
                "next": "mxpi_videodecoder0"
            },
            "mxpi_videodecoder0": {
                "props": {
                    "inputVideoFormat": "H264",
                    "outputImageFormat": "YUV420SP_NV12",
                    "vdecChannelId": "0",
                    "outMode": "1"
                },
                "factory": "mxpi_videodecoder",
                "next": "mxpi_imageresize0"
            },
            "mxpi_imageresize0": {
                "props": {
                    "dataSource": "mxpi_videodecoder0",
                    "resizeHeight": "416",
                    "resizeWidth": "416"
                },
                "factory": "mxpi_imageresize",
                "next": "mxpi_tensorinfer0"
            },
            "mxpi_tensorinfer0": {
                "props": {
                    "dataSource": "mxpi_imageresize0",
                    "modelPath": "../models/yolov3/yolov3_tf_bs1_fp16.om"
                },
                "factory": "mxpi_tensorinfer",
                "next": "mxpi_objectpostprocessor0"
            },
            "mxpi_objectpostprocessor0": {
                "props": {
                    "dataSource": "mxpi_tensorinfer0",
                    "funcLanguage": "c++",
                    "postProcessConfigPath": "../models/yolov3/yolov3_tf_bs1_fp16.cfg",
                    "labelPath": "../models/yolov3/coco.names",
                    "postProcessLibPath": "../../../lib/modelpostprocessors/libyolov3postprocess.so"
                },
                "factory": "mxpi_objectpostprocessor",
                "next": "mxpi_distributor0"
            },
            "mxpi_distributor0": {
                "props": {
                    "dataSource": "mxpi_objectpostprocessor0",
                    "classIds": "0"
                },
                "factory": "mxpi_distributor",
                "next": "mxpi_objectfilter0"
            },
            "mxpi_objectfilter0": {
                "props": {
                    "dataSource": "mxpi_distributor0_0"
                },
                "factory": "mxpi_objectfilter",
                "next": "mxpi_motsimplesort0"
            },
            "mxpi_motsimplesort0": {
                "props": {
                    "dataSourceDetection": "mxpi_objectfilter0"
                },
                "factory": "mxpi_motsimplesort",
                "next": "mxpi_imagecrop0"
            },
            "mxpi_imagecrop0": {
                "props": {
                    "dataSource": "mxpi_objectfilter0",
                    "dataSourceImage": "mxpi_videodecoder0",
                    "resizeHeight": "192",
                    "resizeWidth": "192",
                    "resizeType": "Resizer_KeepAspectRatio_Fit",
                    "paddingType": "Padding_RightDown"
                },
                "factory": "mxpi_imagecrop",
                "next": "mxpi_stackframe0"
            },
            "mxpi_stackframe0": {
                "props": {
                    "visionSource": "mxpi_imagecrop0",
                    "trackSource": "mxpi_motsimplesort0",
                    "frameNum": "5"
                },
                "factory": "mxpi_stackframe",
                "next": "mxpi_tensorinfer1"
            },
            "mxpi_tensorinfer1": {
                "props": {
                    "dataSource": "mxpi_stackframe0",
                    "modelPath": "../models/x3d/x3d_s1.om",
                    "singleBatchInfer": "1",
                    "waitingTime": "250000"
                },
                "factory": "mxpi_tensorinfer",
                "next": "mxpi_classpostprocessor0"
            },
            "mxpi_classpostprocessor0": {
                "props": {
                    "dataSource": "mxpi_tensorinfer1",
                    "postProcessConfigPath": "../models/x3d/x3d_post.cfg",
                    "labelPath": "../models/x3d/kinetics400.names",
                    "postProcessLibPath": "../../../lib/modelpostprocessors/libresnet50postprocess.so"
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
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    streamName = b'detection+action recognition'
    idx = 0
    while idx < args.MAX_COUNT_IDX:
        inferResult = streamManagerApi.GetResult(streamName, 0, 1000000)
        if inferResult is None:
            break
        if inferResult.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                inferResult.errorCode, inferResult.data.decode()))
            break
        retStr = inferResult.data.decode()
        idx += 1
        print(retStr)
    streamManagerApi.DestroyAllStreams()


if __name__ == "__main__":
    if args.TYPE == "main":
        main()
    else:
        sub()
