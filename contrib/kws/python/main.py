#!/usr/bin/env python
# coding=utf-8

"""
Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
Description: Complete Example Implementation of Input Tensor Inference in python.
Author: MindX SDK
Create: 2020
History: NA
"""
from utils import read_conf
from preprocessing import ExtractLogmel
from post_process import infer
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, InProtobufVector, MxProtobufIn, StringVector, MxDataInput

if __name__ == '__main__':
    # init stream manager

    pipeline_path = "../pipeline/crnn_ctc.pipeline"
    streamName = b'kws'
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

    params = read_conf("../data/data.yaml")
    wav_path = "../data/BAC009S0048W0157.wav"
    extract_logmel = ExtractLogmel(max_len=1464, mean_std_path='../data/mean_std.npz')
    feature, feat_real_len = extract_logmel.extract_feature(wav_path,
                                                            params["data"]["num_mel_bins"],
                                                            scale_flag=True)
    tensor = feature[None]

    inPluginId = 0
    tensorPackageList = MxpiDataType.MxpiTensorPackageList()
    tensorPackage = tensorPackageList.tensorPackageVec.add()
    print(tensor.shape)
    array_bytes = tensor.tobytes()
    dataInput = MxDataInput()
    dataInput.data = array_bytes
    tensorVec = tensorPackage.tensorVec.add()
    tensorVec.deviceId = 0
    tensorVec.memType = 0
    for i in tensor.shape:
        tensorVec.tensorShape.append(i)
    tensorVec.dataStr = dataInput.data
    tensorVec.tensorDataSize = len(array_bytes)

    key = "appsrc{}".format(inPluginId).encode('utf-8')
    protobufVec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensorPackageList.SerializeToString()
    protobufVec.push_back(protobuf)

    ret = streamManagerApi.SendProtobuf(streamName, inPluginId, protobufVec)
    if ret < 0:
        print("Failed to send data to stream.")
        exit()

    keyVec = StringVector()
    keyVec.push_back(b'mxpi_tensorinfer0')
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
    res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
    print("output tensor is: ", res)

    # The actual output length of the original data after the model
    seq_len = feat_real_len // 4
    predict_text = infer(res,
                         seq_len,
                         params["data"]["ind2pinyin"],
                         params["data"]["keyword_pinyin_dict"],
                         params["data"]["pinyin2char"])

    print("{}: {}".format(wav_path, predict_text))










