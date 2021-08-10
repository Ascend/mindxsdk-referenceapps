#!/usr/bin/env python
# coding=utf-8

"""
Copyright(C) Huawei Technologies Co.,Ltd. 2012-2021 All rights reserved.

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
import json

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import MxProtobufIn, InProtobufVector, StringVector, StreamManagerApi, MxDataInput, \
   MxBufferInput, MxMetadataInput, MetadataInputVector

# 为SendProtobuf接口准备protobuf_vec参数
def prepare_data():
    vision_list_sample = MxpiDataType.MxpiVisionList()
    vision_vec_sample = vision_list_sample.visionVec.add()
    vision_vec_sample.visionData.dataStr = data_input.data

    protobuf = MxProtobufIn()
    protobuf.key = b'appsrc0'
    protobuf.type = b'MxTools.MxpiVisionList'
    protobuf.protobuf = vision_list_sample.SerializeToString()
    protobuf_vec_sample = InProtobufVector()
    protobuf_vec_sample.push_back(protobuf)
    return protobuf_vec_sample

# 调取GetResultWithUniqueId接口，获取并打印结果
def get_result_with_unique_id_sample(unique_id_sp, receive_stream_name):
    if unique_id_sp < 0:
        message = "Failed to send data to stream."
        print(message)
        exit()

    # Obtain the inference result by specifying streamName and uniqueId.
    infer_result_sample = stream_manager_api.GetResultWithUniqueId(receive_stream_name, unique_id_sp, 3000)
    if infer_result_sample.errorCode != 0:
        print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
            infer_result_sample.errorCode, infer_result_sample.data.decode()))
        exit()
    # print the infer result
    print("result: {}".format(infer_result_sample.data.decode()))

# 调取GetProtobuf接口，获取并打印结果
def get_proto_buffer_sample(receive_key, receive_stream_name, receive_in_plugin_id):
    key_vec_sample = StringVector()
    key_vec_sample.push_back(receive_key)

    # GetProtobuf接口返回结果是MxProtobufOut结构的list
    infer_result_sample = stream_manager_api.GetProtobuf(receive_stream_name, receive_in_plugin_id, key_vec_sample)

    if infer_result_sample.size() == 0:
        print("infer_result is null")
        exit()

    if infer_result_sample[0].errorCode != 0:
        print("GetProtobuf error.errorCode=%d" % (infer_result_sample[0].errorCode))
        exit()

    print("GetProtobuf errorCode=%d" % (infer_result_sample[0].errorCode))
    print("key: {}".format(str(infer_result_sample[0].messageName)))

    # 用MxpiVisionList来接收MxProtobufOut的messageBuf
    result_sample = MxpiDataType.MxpiVisionList()
    result_sample.ParseFromString(infer_result_sample[0].messageBuf)

    print("result: {}".format(result_sample.visionVec[0].visionData.dataStr))


if __name__ == '__main__':
    """
    1. INTERFACE_TYPE 表示接口类型
        1,2,3 代表SendData 接口；
        4,5 代表SendDataWithUniqueId 接口；
        6,7 代表SendProtobuf 接口。
    2. 默认INTERFACE_TYPE为1.
    """
    if len(sys.argv) <= 1:
        INTERFACE_TYPE = 1
    else:
        INTERFACE_TYPE = int(sys.argv[1])

    # 初始化 stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # 通过pipeline配置文件创建 streams
    pipeline = {
        "test": {
            "appsrc0": {
                "props": {
                    "blocksize": "409600"
                },
                "factory": "appsrc",
                "next": "appsink0"
            },
            "appsink0": {
                "props": {
                    "blocksize": "4096000"
                },
                "factory": "appsink"
            }
        }
    }

    pipelineStr = json.dumps(pipeline).encode()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # 构造 stream 输入
    data_input = MxDataInput()
    BUFFER = "Success"
    data_input.data = json.dumps(BUFFER).encode()

    if INTERFACE_TYPE == 1:
        # 执行SendData - GetResult 样例
        stream_name = b'test'
        in_plugin_id = 0
        unique_id = stream_manager_api.SendData(stream_name, in_plugin_id, data_input)

        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()

        # Obtain the inference result by specifying streamName and uniqueId.
        infer_result = stream_manager_api.GetResult(stream_name, unique_id)
        if infer_result.errorCode != 0:
            print("GetResult error. errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))
            exit()
        # 打印推理结果
        print("result1: {}".format(infer_result.data.decode()))

    elif INTERFACE_TYPE == 2:
        # 执行SendData - GetResult 样例
        stream_name = b'test'
        element_name = b'appsrc0'
        unique_id = stream_manager_api.SendData(stream_name, element_name, data_input)
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()
        infer_result = stream_manager_api.GetResult(stream_name, unique_id)
        if infer_result.errorCode != 0:
            print("GetResult error. errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))
            exit()
        # 打印推理结果
        print("result2: {}".format(infer_result.data.decode()))

    elif INTERFACE_TYPE == 3:
        # 执行SendData - GetResult 样例
        stream_name = b'test'
        in_plugin_id = 0
        element_name = b'appsrc0'
        key = b'appsrc0'

        # build senddata data source
        frame_info = MxpiDataType.MxpiFrameInfo()
        frame_info.frameId = 0
        frame_info.channelId = 0

        vision_list = MxpiDataType.MxpiVisionList()
        vision_vec = vision_list.visionVec.add()
        vision_vec.visionData.dataStr = data_input.data

        # 构建MxBufferInput对象
        buffer_input = MxBufferInput()
        buffer_input.mxpiFrameInfo = frame_info.SerializeToString()
        buffer_input.mxpiVisionInfo = vision_vec.SerializeToString()
        buffer_input.data = data_input.data

        # 构建MxMetadataInput对象
        metedata_input = MxMetadataInput()
        metedata_input.dataSource = b'appsrc0'
        metedata_input.dataType = b"MxTools.MxpiVisionList"
        metedata_input.serializedMetadata = vision_list.SerializeToString()

        # 构建MetadataInputVector对象
        metedata_vec = MetadataInputVector()
        metedata_vec.push_back(metedata_input)

        error_code = stream_manager_api.SendData(stream_name, element_name, metedata_vec, buffer_input)
        if error_code < 0:
            print("Failed to send data to stream.")
            exit()
        # 构建GetResult接口参数
        data_source_vector = StringVector()
        data_source_vector.push_back(b'appsrc0')

        infer_result = stream_manager_api.GetResult(stream_name, b'appsink0', data_source_vector)

        if infer_result.errorCode != 0:
            print("GetResult failed")
            exit()

        if infer_result.bufferOutput.data is None:
            print("bufferOutput nullptr")
            exit()
        # 打印结果
        print("result3: {}".format(infer_result.bufferOutput.data.decode()))

    elif INTERFACE_TYPE == 4:
        # 执行SendDataWithUniqueId - GetResultWithUniqueId 样例

        stream_name = b'test'
        in_plugin_id = 0

        unique_id = stream_manager_api.SendDataWithUniqueId(stream_name, in_plugin_id, data_input)

        get_result_with_unique_id_sample(unique_id, stream_name)

    elif INTERFACE_TYPE == 5:
        # 执行SendDataWithUniqueId - GetResultWithUniqueId 样例

        stream_name = b'test'
        element_name = b'appsrc0'

        unique_id = stream_manager_api.SendDataWithUniqueId(stream_name, element_name, data_input)

        get_result_with_unique_id_sample(unique_id, stream_name)

    elif INTERFACE_TYPE == 6:
        # 执行SendProtobuf - GetProtobuf 样例
        stream_name = b'test'
        in_plugin_id = 0
        key = b'appsrc0'

        protobuf_vec = prepare_data()
        error_code = stream_manager_api.SendProtobuf(stream_name, in_plugin_id, protobuf_vec)
        if error_code < 0:
            print("Failed to send data to stream.")
            exit()
        get_proto_buffer_sample(key, stream_name, in_plugin_id)
    elif INTERFACE_TYPE == 7:
        # 执行SendProtobuf - GetProtobuf 样例
        stream_name = b'test'
        element_name = b'appsrc0'
        in_plugin_id = 0
        key = b'appsrc0'

        protobuf_vec = prepare_data()
        error_code = stream_manager_api.SendProtobuf(stream_name, element_name, protobuf_vec)
        if error_code < 0:
            print("Failed to send data to stream.")
            exit()
        get_proto_buffer_sample(key, stream_name, in_plugin_id)

    else:
        print("请选择正确的类型")

    # 销毁 streams
    stream_manager_api.DestroyAllStreams()
