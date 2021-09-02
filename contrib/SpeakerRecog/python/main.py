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
import os
import utils
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
    test_wav_dir = "../test_wav"
    wav_path_list = utils.get_all_type_paths(test_wav_dir, ".wav")
    if len(wav_path_list) == 0:
        print('There is no wav audio in {}!'.format(test_wav_dir))
        print('Please change the audio in wav format!')
        exit()
    wav_path = wav_path_list[0]
    wav_name = os.path.basename(wav_path).split(".")[0]
    extract_logmel = ExtractLogmel(padded_type="copy")
    # extract feature
    # the dimension of log_mel_spectrogram
    feature_dim = 64
    feature, feat_real_len = extract_logmel.extract_feature(wav_path,
                                                            feat_dim=feature_dim,
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
