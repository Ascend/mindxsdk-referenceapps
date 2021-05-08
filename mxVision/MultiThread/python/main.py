#!/usr/bin/env python
# coding=utf-8

"""
Copyright 2020 Huawei Technologies Co., Ltd

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

import sys
from StreamManagerApi import MxDataInput, StreamManagerApi
import threading


def SendAndGetData(streamManagerApi, streamName, threadId, dataInput):
    for i in range(5):
        print("Start to send data, threadId = %d" % (threadId))
        ret = streamManagerApi.SendData(streamName, 0, dataInput)
        if ret != 0:
            print("Failed to send data to stream.")
            exit()

        # Obtain the inference result by specifying streamName and uniqueId.
        inferResult = streamManagerApi.GetResult(streamName, 0, 15000)
        if inferResult.errorCode != 0:
            print("GetResult error. errorCode=%d" % (inferResult.errorCode))
            exit()
        print("End to get data, threadId = %d, result = %s" % (threadId, inferResult.data))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        fileName = sys.argv[1]
    else:
        fileName = "./EasyStream.pipeline"
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(fileName, 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    dataInput = MxDataInput()
    with open("../picture/test.jpg", 'rb') as f:
        dataInput.data = f.read()

    # Inputs data to a specified stream based on streamName.
    streamName = b'detection0'
    # Multi thread

    thread1_1 = threading.Thread(target=SendAndGetData, name='SendAndGetData0', args=(streamManagerApi,
                                                                                      streamName, 1, dataInput))
    thread1_2 = threading.Thread(target=SendAndGetData, name='SendAndGetData1', args=(streamManagerApi,
                                                                                      streamName, 2, dataInput))
    streamName = b'detection1'
    thread2_1 = threading.Thread(target=SendAndGetData, name='SendAndGetData2', args=(streamManagerApi,
                                                                                      streamName, 3, dataInput))
    thread2_2 = threading.Thread(target=SendAndGetData, name='SendAndGetData3', args=(streamManagerApi,
                                                                                      streamName, 4, dataInput))
    streamName = b'detection2'
    thread3_1 = threading.Thread(target=SendAndGetData, name='SendAndGetData4', args=(streamManagerApi,
                                                                                      streamName, 5, dataInput))
    thread3_2 = threading.Thread(target=SendAndGetData, name='SendAndGetData5', args=(streamManagerApi,
                                                                                      streamName, 6, dataInput))
    thread1_1.start()
    thread1_2.start()
    thread2_1.start()
    thread2_2.start()
    thread3_1.start()
    thread3_2.start()

    thread1_1.join()
    thread1_2.join()
    thread2_1.join()
    thread2_2.join()
    thread3_1.join()
    thread3_2.join()
    # destroy streams
    streamManagerApi.DestroyAllStreams()
