#!/usr/bin/env python
# coding=utf-8

# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
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

# import StreamManagerApi.py
from StreamManagerApi import *
import os
if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
    #   exit()

    # create streams by pipeline config file
    with open("./pipeline/Sample.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))

    file_handle = open('img_result.txt', 'w')
    file_handle2 = open('test_full.txt', 'r')
    while 1:
        # Construct the input of the stream
        dataInput = MxDataInput()
        line = file_handle2.readline()
        img_path = line[0:34]
        with open(img_path, 'rb') as f:
            dataInput.data = f.read()
     
        # Inputs data to a specified stream based on streamName.
        streamName = b'classification+detection'
        inPluginId = 0
        uniqueId = streamManagerApi.SendDataWithUniqueId(streamName, inPluginId, dataInput)
        if uniqueId < 0:
            print("Failed to send data to stream.")

        # Obtain the inference result by specifying streamName and uniqueId.
        inferResult = streamManagerApi.GetResultWithUniqueId(streamName, uniqueId, 3000)
        if inferResult.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
              inferResult.errorCode, inferResult.data.decode()))
        # save the confidence
        img_list = []
        # print the infer result
        print(inferResult.data.decode())
        dict_all = inferResult.data.decode()
        
        begin = 0
        end = len(dict_all)
        index = 0
        for i in range(40):
            # find confidence in dict 
            loc = dict_all.find('confidence', begin + index, end)
            # next confidence
            index = loc + 12
            img_list.append(int(dict_all[loc + 12]))
        file_handle.write(img_path + ' ')
        file_handle.write(str(img_list).replace("[", "").replace("]", "").replace(",", " "))
        file_handle.write('\n')
    # destroy streams
    file_handle.close()
    streamManagerApi.DestroyAllStreams()
