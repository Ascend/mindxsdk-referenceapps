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

import stat
import json
import os
import time
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

if __name__ == '__main__':
    FLAGS = os.O_RDWR | os.O_APPEND | os.O_CREAT
    MODES = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRWXU | stat.S_IEXEC
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    with open("./pipeline/passengerflowestimation.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    STREAMNAME = b'passengerflowestimation_pipline'
    # save the result
    FRAMEID = 0
    FRAMENUM = 1300
    startTime = time.time()
    with os.fdopen(os.open('result.h264', FLAGS, MODES), 'ab+') as f:
        while FRAMEID < FRAMENUM:
            FRAMEID += 1
            infer_result = streamManagerApi.GetResult(STREAMNAME, 0, 10000)
            if infer_result.data == b'[1002][Internal error] ':
                print("Error! cannot find video source!")
                break
            f.write(infer_result.data)
    endTime = time.time()
    rate = FRAMENUM/(endTime - startTime)
    print("Average_framerate:", rate)
    f.close()
    streamManagerApi.DestroyAllStreams()
