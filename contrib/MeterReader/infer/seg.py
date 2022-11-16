#!/usr/bin/env python
# coding=utf-8

# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
import json
import getopt
import stat
import cv2
import numpy as np

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

PIPELINE = '../pipeline/deeplabv3/seg.pipeline'


def get_args():
    argv = sys.argv[1:]
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg

    return inputfile


if __name__ == '__main__':

    input_file_name = get_args()

    steammanager_api = StreamManagerApi()

    ret = steammanager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    MODES = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(PIPELINE, os.O_RDONLY, MODES), 'rb') as f:
        pipeline_str = f.read()
    ret = steammanager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Create Stream Fail !!! ret=%s" % str(ret))
        exit()
    input_data = MxDataInput()

    if os.path.exists(input_file_name) != 1:
        print("The test image does not exist. Exit.")
        exit()
    with os.fdopen(os.open(input_file_name, os.O_RDONLY, stat.S_IWUSR | stat.S_IRUSR), 'rb') as f:
        input_data.data = f.read()
        
    STREAM_NAME = b'seg'
    IN_PLUGIN_ID = 0
    uId = steammanager_api.SendData(STREAM_NAME, IN_PLUGIN_ID, input_data)
    if uId < 0:
        print("Failed to send data to stream.")
        exit()

    keyVec = StringVector()
    keyVec.push_back(b"mxpi_process3")

    infer = steammanager_api.GetResult(STREAM_NAME, b'appsink0', keyVec)
    print("-------------------------")
    result = MxpiDataType.MxpiClass()
    result.ParseFromString(infer.metadataVec[0].serializedMetadata)
    print(f"seg_ans {result.confidence}")

    # destroy streams
    steammanager_api.DestroyAllStreams()
