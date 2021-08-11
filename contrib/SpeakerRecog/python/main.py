#!/usr/bin/env python
# coding=utf-8

"""
Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
Description: Complete Example Implementation of Input Tensor Inference in python.
Author: MindX SDK
Create: 2020
History: NA
"""
import os

from preprocessing import ExtractLogmel
from post_process import speaker_recognition
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, InProtobufVector, MxProtobufIn, StringVector, MxDataInput

if __name__ == '__main__':
    # init stream manager
    # pipeline file path
    pipeline_path = "../pipeline/SpeakerRecog.pipeline"
    streamName = b'SpeakerRecog'
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(pipeline_path, 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    # voice_print_library
    voice_print_library_path = "../voice_print_library"
    # test wav
    wav_path = "../test_wav/BAC009S0766W0422.wav"
    wav_name = os.path.basename(wav_path).split(".")[0]
    extract_logmel = ExtractLogmel(padded_type="copy")
    # extract feature
    feature, feat_real_len = extract_logmel.extract_feature(wav_path,
                                                            feat_dim=64,
                                                            scale_flag=False)
    # (64, 1000) to (1, 64, 1000)
    tensor = feature[None]

    inPluginId = 0
    tensorPackageList = MxpiDataType.MxpiTensorPackageList()
    tensorPackage = tensorPackageList.tensorPackageVec.add()
    # add feature data begin
    array_bytes = tensor.tobytes()
    dataInput = MxDataInput()
    dataInput.data = array_bytes
    tensorVec = tensorPackage.tensorVec.add()
    tensorVec.deviceId = 0
    tensorVec.memType = 0
    for i in tensor.shape:
        tensorVec.tensorShape.append(i)
    tensorVec.dataStr = dataInput.data
    # compute the number of bytes of feature data
    tensorVec.tensorDataSize = len(array_bytes)
    # add feature data end

    key = "appsrc{}".format(inPluginId).encode('utf-8')
    protobufVec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensorPackageList.SerializeToString()
    protobufVec.push_back(protobuf)

    uniqueId = streamManagerApi.SendProtobuf(streamName, inPluginId, protobufVec)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    keyVec = StringVector()
    keyVec.push_back(b'mxpi_tensorinfer0')
    # get inference result
    inferResult = streamManagerApi.GetProtobuf(streamName, 0, keyVec)

    if inferResult.size() == 0:
        print("inferResult is null")
        exit()
    if inferResult[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" % (
            inferResult[0].errorCode))
        exit()
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(inferResult[0].messageBuf)
    # convert the inference result to Numpy array
    # speaker embedding
    res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
    # post_processing
    speaker_recognition(res, wav_name, voice_print_library_path)














