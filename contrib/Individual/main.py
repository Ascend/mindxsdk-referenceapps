
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

from StreamManagerApi import StreamManagerApi,MxDataInput

if __name__ == '__main__':
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))

    # create streams by pipeline config file
    with open("./pipeline/DectetionAndAttr.pipeline", 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))

    # Construct the input of the stream
    data_input = MxDataInput()

    # example
    with open("./test.jpg", 'rb') as f:
        data_input.data = f.read()

    # Inputs data to a specified stream based on streamName.
    stream_name = b'classification+detection'
    inplugin_id = 0
    unique_id = stream_manager_api.SendDataWithUniqueId(stream_name, inplugin_id, data_input)
    if unique_id < 0:
        print("Failed to send data to stream.")

    # Obtain the inference result by specifying streamName and uniqueId.
    infer_result = stream_manager_api.GetResultWithUniqueId(stream_name, unique_id, 3000)
    if infer_result.errorCode != 0:
        print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
            infer_result.errorCode, infer_result.data.decode()))

    # print the infer result
    print(infer_result.data.decode())

    # destroy streams
    stream_manager_api.DestroyAllStreams()
