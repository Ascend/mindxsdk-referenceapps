# Copyright 2021 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os
import cv2
import numpy as np

from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
import MxpiDataType_pb2 as MxpiDataType


if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./collision.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Inputs data to a specified stream based on streamName.
    STREAM_NAME = b'collision'

    fd = os.open( './out_collision.h264' , os.O_RDWR | os.O_CREAT , 640)
    fo = os.fdopen(fd, "wb")
    T = 0
    while True:
        # Obtain the inference result by specifying streamName and uniqueId.
        infer_result = streamManagerApi.GetResult(STREAM_NAME, 0, 300000)
        fo.write(infer_result.data)
        T = T+1
        print(T)
        # the toal frames of video
        if T > 100:
            print('ending')
            break
    
    fo.close()
    # destroy streams
    streamManagerApi.DestroyAllStreams()