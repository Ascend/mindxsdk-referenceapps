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

import json
import time
from stream_manager_api import *
import MxpiOSDType_pb2 as MxpiOSDType
from google.protobuf.json_format import *

if __name__ == '__main__':
    # init stream manager
    stream_manager_api = stream_manager_api()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("../pipeline/SampleOsd.pipeline", 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    data_input = Mxdata_input()
    with open("test.jpg", 'rb') as f:
        data_input.data = f.read()

    # Send image.
    stream_name = b'encoder'
    in_plugin_id = 0
    ret = stream_manager_api.SendData(stream_name, in_plugin_id, data_input)
    if ret < 0:
        print("Failed to send data to stream.")
        exit()

    # Send osd instances protobuf.
    with open("ExternalOsdInstances.json", "r") as f:
        message_json = json.load(f)
    print(message_json)
    in_plugin_id = 1
    osd_instances_list = MxpiOSDType.Mxpiosd_instances_list()
    osd_instances_list = ParseDict(message_json, osd_instances_list)

    protobuf_vec = Inprotobuf_vector()
    protobuf = MxProtobufIn()
    protobuf.key = b'appsrc1'
    protobuf.type = b'MxTools.Mxpiosd_instances_list'
    protobuf.protobuf = osd_instances_list.SerializeToString()
    protobuf_vec.push_back(protobuf)
    ret = stream_manager_api.SendProtobuf(stream_name, in_plugin_id, protobuf_vec)
    if ret < 0:
        print("Failed to send protobuf to stream.")
        exit()

    time.sleep(2)

    # destroy streams
    stream_manager_api.DestroyAllStreams()
