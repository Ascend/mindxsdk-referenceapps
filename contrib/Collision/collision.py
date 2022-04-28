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
    streamName = b'collision'
    inPluginId = 0

    f=open('./out_collision.h264','wb')
    t=0
    while True:
        # Obtain the inference result by specifying streamName and uniqueId.
        infer_result = streamManagerApi.GetResult(streamName, 0, 300000)
        f.write(infer_result.data)  
        t=t+1
        print(t)
        if t>350:  #the toal frames of video
          print('ending')
          break
    
    f.close()
    # destroy streams
    streamManagerApi.DestroyAllStreams()