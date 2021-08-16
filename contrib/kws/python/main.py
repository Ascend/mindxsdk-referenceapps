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
# limitations under the License
from utils import read_conf
import utils
from preprocessing import ExtractLogmel
from post_process import infer
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, InProtobufVector, MxProtobufIn, StringVector, MxDataInput

if __name__ == '__main__':
    # init stream manager
    # pipeline file's path
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
    # Read parameter file
    params = read_conf("../data/data.yaml")
    # test wav
    wav_path_list = utils.get_all_type_paths("../data", ".wav")
    wav_path = wav_path_list[0]
    # extract feature
    max_frames = 1464
    extract_logmel = ExtractLogmel(max_len=max_frames, mean_std_path='../data/mean_std.npz')
    feature, feat_real_len = extract_logmel.extract_feature(wav_path,
                                                            params["data"]["num_mel_bins"],
                                                            scale_flag=True)
    # (80, 1464) to (1, 80, 1464)
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
    res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
    print("output tensor is: ", res)
    # post_processing
    # The actual output length of the original data after the model
    pool_factor = 4
    seq_len = feat_real_len // pool_factor
    # decoding
    predict_text = infer(res,
                         seq_len,
                         params["data"]["num_classes"],
                         params["data"]["ind2pinyin"],
                         params["data"]["keyword_pinyin_dict"],
                         params["data"]["pinyin2char"])
    # output the inference result
    print("{}: {}".format(wav_path, predict_text))
