#!/usr/bin/env python
# coding=utf-8

"""

 Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.

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

from StreamManagerApi import StreamManagerApi, StringVector

if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./pipeline/plugin_alone.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Inputs data to a specified stream based on streamName.
    streamName = b'detection+tracking'
    inPluginId = 0

    retStr = ''
    key = b'mxpi_pluginalone0'
    keyVec = StringVector()
    keyVec.push_back(key)
    while True:
        # Obtain the inference result by specifying streamName and uniqueId.
        inferResult = streamManagerApi.GetResult(streamName, 0, 10000)
        if inferResult is None:
            break
        if inferResult.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                inferResult.errorCode, inferResult.data.decode()))
            break
        retStr = inferResult.data.decode()
        print(retStr)

    # destroy streams
    streamManagerApi.DestroyAllStreams()