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

from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

import sys
sys.path.append("../../proto")
import mxpiSampleProto_pb2 as mxpiSampleProto

if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("../pipeline/Sample_proto.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    dataInput = MxDataInput()
    with open("test.jpg", 'rb') as f:
        dataInput.data = f.read()

    # The following is how to set the dataInput.roiBoxs
    """
    roiVector = RoiBoxVector()
    roi = RoiBox()
    roi.x0 = 100
    roi.y0 = 100
    roi.x1 = 200
    roi.y1 = 200
    roiVector.push_back(roi)
    dataInput.roiBoxs = roiVector
    """

    # Inputs data to a specified stream based on streamName.
    streamName = b'classification+detection'
    inPluginId = 0
    uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    # get protobuf with custom
    key_vec = StringVector()
    # choose which metadata to be got.In this case we use the custom "mxpi_sampleproto"
    key_vec.push_back(b"mxpi_sampleproto")
    infer_result = streamManagerApi. \
        GetProtobuf(streamName, inPluginId, key_vec)
    if infer_result.size() == 0:
        print("infer_result is null")
        exit()
    if infer_result[0].errorCode != 0:
        print("infer_result error. \
                errorCode=%d" % (infer_result[0].errorCode))
        exit()

    # print the infer result
    print("GetProtobuf errorCode=%d" % (infer_result[0].errorCode))
    print("KEY: {}".format(str(infer_result[0].messageName)))

    result_protolist = mxpiSampleProto.MxpiSampleProtoList()
    result_protolist.ParseFromString(infer_result[0].messageBuf)
    print("result: {}".format(
        result_protolist.sampleProtoVec[0].stringSample))

    # destroy streams
    streamManagerApi.DestroyAllStreams()
