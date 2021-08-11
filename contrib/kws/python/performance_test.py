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
import time
import numpy as np
from utils import read_conf, read_info
from preprocessing import ExtractLogmel
from post_process import infer
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, InProtobufVector, MxProtobufIn, StringVector, MxDataInput

if __name__ == "__main__":
    params = read_conf("..data/data.yaml")
    data_info_dir = os.path.join(params["data"]["data_info_dir"], "test")
    info_path = os.path.join(data_info_dir, "%s.%s.keyword" % (params["data"]["name"], "test"))
    info_jsons = read_info(info_path, min_duration=0., max_duration=100)
    info_path = os.path.join(data_info_dir, "%s.%s.non_keyword" % (params["data"]["name"], "test"))
    info_jsons.extend(read_info(info_path, min_duration=0., max_duration=100))
    wav_list = list()
    time_list = list()
    label_list = list()
    text_list = list()
    for _, json_line in enumerate(info_jsons):
        wav_list.append(json_line["wav_path"])
        time_list.append(json_line["duration"])
        label_list.append(json_line["label"])
        text_list.append(json_line["text"])
    total_time_duration = sum(time_list)
    detect_result = []

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

    print("========infer start=========")
    extract_logmel = ExtractLogmel(max_len=1464, mean_std_path='../data/mean_std.npz')
    begin = time.time()
    for idx, wav_path in enumerate(wav_list):
        feature, feat_real_len = extract_logmel.extract_feature(wav_path,
                                                                feat_dim=80,
                                                                scale_flag=True)
        tensor = feature[None]
        inPluginId = 0
        tensorPackageList = MxpiDataType.MxpiTensorPackageList()
        tensorPackage = tensorPackageList.tensorPackageVec.add()
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
        # The actual output length of the original data after the model
        seq_len = feat_real_len // 4
        predict_text = infer(res,
                             seq_len,
                             params["data"]["ind2pinyin"],
                             params["data"]["keyword_pinyin_dict"],
                             params["data"]["pinyin2char"])
        detect_result.append(predict_text)
    end = time.time()
    print("========infer end=========")
    keyword_list = params["data"]["keyword_list"]
    total_keywords_num = [0 for _ in keyword_list]
    detect_keywords_num = [0 for _ in keyword_list]
    correct_detect_keywords_num = [0 for _ in keyword_list]
    for idx, word in enumerate(keyword_list):
        for index, item in enumerate(detect_result):
            target_num = len(text_list[index].split(word))-1
            detect_num = len(item.split(word))-1
            if detect_num <= target_num:
                correct_num = detect_num
            else:
                correct_num = target_num
            total_keywords_num[idx] += target_num
            detect_keywords_num[idx] += detect_num
            correct_detect_keywords_num[idx] += correct_num
    total_count = sum(total_keywords_num)
    detect_count = sum(detect_keywords_num)
    correct_count = sum(correct_detect_keywords_num)
    ave_frr = (total_count - correct_count) / total_count if total_count > 0 else 1
    scale_factor = 10
    ave_far = (detect_count - correct_count) / (len(keyword_list) * (total_time_duration / 3600) * scale_factor)
    print("{} samples' total time{}".format(len(wav_list), total_time_duration))
    print("{} samples' infer time{}".format(len(wav_list), end-begin))
    print("FRR: {}".format(ave_frr))
    print("FAR: {}".format(ave_far))





