#!/usr/bin/env python
# coding=utf-8

"""
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
"""

import codecs
import csv
import time
import os
import numpy as np

from tokenizer import Tokenizer

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, StringVector, MxProtobufIn, InProtobufVector

maxlen = 498
tensor_length = 500
float32_bytes = 4
all_sample_number = 2000
token_dict = {}
label = ["消极", "积极", "中性"]

class OurTokenizer(Tokenizer):
    """
    tokenizer text to ID
    """
    def _tokenize(self, content):
        R = []
        for c in content:
            if c in self._token_dict:
                R.append(c)
            else:
                # The remaining characters are [UNK]
                R.append('[UNK]')
        return R


def save_to_file(file_name, contents):
    """
    save prediction label to file
    """
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()


def preprocess(input_text):
    """
    tokenizer text to ID. Fill with 0 if the text  is not long than maxlen
    """
    tokenizer = OurTokenizer(token_dict)

    # tokenize
    input_text = input_text[:maxlen]
    x1, x2 = tokenizer.encode(first=input_text)

    # if the length of the text is less than maxLen, padding 0
    x1 = x1 + [0] * (tensor_length - len(x1)) if len(x1) < tensor_length else x1
    x2 = x2 + [0] * (tensor_length - len(x2)) if len(x2) < tensor_length else x2

    x1 = np.ascontiguousarray(x1, dtype='float32')
    x2 = np.ascontiguousarray(x2, dtype='float32')

    x1 = np.expand_dims(x1, 0)
    x2 = np.expand_dims(x2, 0)
    return x1, x2


if __name__ == '__main__':
    streamManagerApi = StreamManagerApi()
    # init StreamManager
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

     # creat streams by pipeline file
    if os.path.exists("pipeline/sentiment_analysis.pipeline") != 1:
        print("pipeline file does not exist")
        exit()
    with open("pipeline/sentiment_analysis.pipeline", 'rb') as f:
        pipelineStr = f.read()

    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # read the vocab text
    if os.path.exists("data/vocab.txt") != 1:
        print("The vocab.txt does not exist")
        exit()
    with codecs.open("data/vocab.txt", 'r', 'utf-8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    if os.path.exists("data/test.csv") != 1:
        print("test.csv does not exist")
        exit()

    # read the input text
    csv_reader = csv.reader(open("data/test.csv"))
    result_list = [0, 0, 0]
    real_list = [0, 0, 0]
    count = 0
    time_start = time.time()
    for line in csv_reader:
        real_label = line[0]
        text = line[1]
        # preprocess the data
        X1, X2 = preprocess(text)

        streamName = b'sentiment_analysis'
        inPluginId = 0
        protobuf_vec = InProtobufVector()

        mxpi_tensor_package_list = MxpiDataType.MxpiTensorPackageList()
        tensor_package_vec = mxpi_tensor_package_list.tensorPackageVec.add()

        # set the first tensor input
        tensorVec = tensor_package_vec.tensorVec.add()
        tensorVec.memType = 1
        tensorVec.deviceId = 1
        tensorVec.tensorDataSize = int(tensor_length * float32_bytes)
        tensorVec.tensorDataType = 0  # float32
        for i in X1.shape:
            tensorVec.tensorShape.append(i)
        tensorVec.dataStr = X1.tobytes()

        # set the second tensor input
        tensorVec2 = tensor_package_vec.tensorVec.add()
        tensorVec2.memType = 1
        tensorVec2.deviceId = 1
        tensorVec2.tensorDataSize = int(tensor_length * float32_bytes)
        tensorVec2.tensorDataType = 0  # float32
        for i in X2.shape:
            tensorVec2.tensorShape.append(i)
        tensorVec2.dataStr = X2.tobytes()

        protobuf = MxProtobufIn()
        protobuf.key = b'appsrc0'
        protobuf.type = b'MxTools.MxpiTensorPackageList'
        protobuf.protobuf = mxpi_tensor_package_list.SerializeToString()
        protobuf_vec.push_back(protobuf)
        # send data into the stream according to the stream name
        uniqueId = streamManagerApi.SendProtobuf(streamName, inPluginId, protobuf_vec)

        if uniqueId < 0:
            print("Failed to send data to stream.")
            exit()

        key_vec = StringVector()
        key_vec.push_back(b'mxpi_classpostprocessor0')

        # get inference result
        infer = streamManagerApi.GetResult(streamName, b'appsink0', key_vec)
        infer_result = infer.metadataVec[0]

        if infer_result.errorCode != 0:
            print("GetResult error. errorCode=%d ,errorMsg=%s" % (
                infer_result.errorCode, infer_result.errorMsg))
            exit()

        # get data from infer_result
        result = MxpiDataType.MxpiClassList()
        result.ParseFromString(infer_result.serializedMetadata)
        result_label = result.classVec[0].className

        print("第"+str(count + 1)+"个样本")
        if result_label == real_label:
            result_index = label.index(result_label)
            result_list[result_index] += 1
        real_index = label.index(real_label)
        real_list[real_index] += 1
        count += 1

    # destroy streams
    streamManagerApi.DestroyAllStreams()
    time_end = time.time()
    time_average_cost = (time_end - time_start) / all_sample_number
    print('time cost:', str(format(time_average_cost, '.4f')), 's')
    # save and print accuracy
    f = open("out/accuracy.txt", "w")

    i = 0
    for result, real in zip(result_list, real_list):
        accuracy = result / real
        f.write(label[i] + "类的精确度：" + str(format(accuracy, '.4f')) + "\n")
        print(label[i] + "类的精确度：" + str(format(accuracy, '.4f')))
        i += 1

    all_accuracy = sum(result_list) / sum(real_list)
    f.write("全部类别的精确度：" + str(format(all_accuracy, '.4f')) + "\n")
    print("全部类别的精确度：" + str(format(all_accuracy, '.4f')))
    f.close()