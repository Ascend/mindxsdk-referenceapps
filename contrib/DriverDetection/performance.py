# coding=utf-8

"""
Copyright 2021 Huawei Technologies Co., Ltd

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

import time
import sys
from StreamManagerApi import StreamManagerApi, StringVector

if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./pipeline/dirver-detection.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Inputs data to a specified stream based on streamName.
    streamName = b'im_resnet50'
    inPluginId = 0
    total_frame = 0
    st_frame = 0
   
    time_start = time.time() #开始计时
    while True:        
        # Obtain the inference result by specifying streamName and uniqueId.
        inferResult = streamManagerApi.GetResult(streamName, inPluginId, 10000)
        end = time.time()

        if inferResult is None:
            break
        if inferResult.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                inferResult.errorCode, inferResult.data.decode()))
            break
        total_frame = total_frame + 1
        if (int(end - time_start) == 10):
            fps = total_frame / 10
            print("fps:",fps)
            total_frame = 0
            time_start = time.time()
            continue


    # destroy streams
    streamManagerApi.DestroyAllStreams()
