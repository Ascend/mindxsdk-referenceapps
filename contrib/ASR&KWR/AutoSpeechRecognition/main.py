# Copyright 2021 Huawei Technologies Co., Ltd
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

import os
import time
import stat
import numpy as np
import MxpiDataType_pb2 as MxpiDataType

from StreamManagerApi import StreamManagerApi, InProtobufVector, MxProtobufIn, StringVector
from post_process import TextFeaturizer


def levenshtein(u, v):
    prev = None
    curr = [0] + list(range(1, len(v) + 1))
    prev_ops = None
    curr_ops = [(0, 0, i) for i in range(len(v) + 1)]
    for x in range(1, len(u) + 1):
        prev, curr = curr, [x] + ([None] * len(v))
        prev_ops, curr_ops = curr_ops, [(0, x, 0)] + ([None] * len(v))
        for y in range(1, len(v) + 1):
            delcost = prev[y] + 1
            addcost = curr[y - 1] + 1
            subcost = prev[y - 1] + int(u[x - 1] != v[y - 1])
            curr[y] = min(subcost, delcost, addcost)
            if curr[y] == subcost:
                (n_s, n_d, n_i) = prev_ops[y - 1]
                curr_ops[y] = (n_s + int(u[x - 1] != v[y - 1]), n_d, n_i)
            elif curr[y] == delcost:
                (n_s, n_d, n_i) = prev_ops[y]
                curr_ops[y] = (n_s, n_d + 1, n_i)
            else:
                (n_s, n_d, n_i) = curr_ops[y - 1]
                curr_ops[y] = (n_s, n_d, n_i + 1)
    return curr[len(v)] / float(len(v)), curr_ops[len(v)]

if __name__ == "__main__":

    # data type switch
    DATA_RAW = True
    STREAM_NAME = b'speech_recognition'
    TENSOR_KEY = b'appsrc0'
    FLAGS = os.O_RDWR | os.O_CREAT  
    MODES = stat.S_IWUSR | stat.S_IRUSR
    inPluginId, cost_all, wav_count = 0, 0, 0

    # get the path to the current directory
    cwd_path = os.getcwd()
    pipeline_path = os.path.join(cwd_path, "pipeline/am_lm.pipeline")

    streamManager = StreamManagerApi()
    ret = streamManager.InitManager()
    if ret != 0:
        print("Failed to init stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(pipeline_path, 'rb') as f:
        pipeline_str = f.read()
    ret = streamManager.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    path = os.path.join(cwd_path, "data/sample")
    dirs = os.listdir(path)

    wer_s, wer_i, wer_d, wer_n = 0, 0, 0, 0
    k = 0
    text_list = []
    # if data is wav file
    if DATA_RAW is True:
        # not needed if data is numpy file
        from pre_process import make_model_input
        for file in dirs:
            name = file.split('.')[0]
            if file.endswith(".wav"):                
                wav_file_path = os.path.join(cwd_path + "/data/sample/", file)
                feat_data, len_data = make_model_input([wav_file_path])
                np.save(os.path.join(cwd_path, "data/npy/feat_data", name +  "_feat"), feat_data) 
                np.save(os.path.join(cwd_path, "data/npy/len_data", name + "_len"), len_data)
                protobuf_vec = InProtobufVector()
                mxpi_tensor_package_list = MxpiDataType.MxpiTensorPackageList()
                tensor_package_vec = mxpi_tensor_package_list.tensorPackageVec.add()

                # add feature data #begin
                tensorVec = tensor_package_vec.tensorVec.add()
                tensorVec.memType = 1
                tensorVec.deviceId = 0
                # Compute the number of bytes of feature data.
                tensorVec.tensorDataSize = int(
                    feat_data.shape[1]*feat_data.shape[2]*4)
                tensorVec.tensorDataType = 0  # float32
                for i in feat_data.shape:
                    tensorVec.tensorShape.append(i)
                tensorVec.dataStr = feat_data.tobytes()
                # add feature data #end

                # add length data #begin
                tensorVec2 = tensor_package_vec.tensorVec.add()
                tensorVec2.memType = 1
                tensorVec2.deviceId = 0
                # Compute the number of bytes of length data.
                # int(4)  4: btyes of int32
                tensorVec2.tensorDataSize = int(4)
                tensorVec2.tensorDataType = 3  # int32
                for i in len_data.shape:
                    tensorVec2.tensorShape.append(i)
                tensorVec2.dataStr = len_data.tobytes()
                # add length data #end

                protobuf = MxProtobufIn()
                protobuf.key = TENSOR_KEY
                protobuf.type = b'MxTools.MxpiTensorPackageList'
                protobuf.protobuf = mxpi_tensor_package_list.SerializeToString()
                protobuf_vec.push_back(protobuf)
                start = time.time()
                unique_id = streamManager.SendProtobuf(
                    STREAM_NAME, inPluginId, protobuf_vec)
                if unique_id < 0:
                    print("Failed to send data to stream.")
                    exit()

                key_vec = StringVector()
                key_vec.push_back(b'mxpi_tensorinfer1')
                # get inference result
                infer_result = streamManager.GetProtobuf(
                    STREAM_NAME, inPluginId, key_vec)
                if infer_result.size() == 0:
                    print("infer_result is null")
                    exit()
                if infer_result[0].errorCode != 0:
                    print("GetProtobuf error. errorCode=%d" % (
                        infer_result[0].errorCode))
                    exit()

                result = MxpiDataType.MxpiTensorPackageList()
                result.ParseFromString(infer_result[0].messageBuf)

                # converts the inference result to a NumPy array
                ids = np.frombuffer(
                    result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.int32)
                end = time.time()
                cost_all += (end - start)
                # decode
                lm_tokens_path = os.path.join(cwd_path, "data/lm_tokens.txt")
                text_featurizer = TextFeaturizer(lm_tokens_path)
                text = text_featurizer.deocde_without_start_end(ids)
                # convert list to string and print recognition result
                text_list.append(''.join(name)+' '+''.join(text)+'\n')
                wav_count += 1


    else:
        feat_path = os.path.join(cwd_path, "data/npy/feat_data")
        feat_path_list = os.listdir(feat_path)
        len_path = os.path.join(cwd_path, "data/npy/len_data")
        len_path_list = os.listdir(feat_path)
        for file in feat_path_list:
            name = file.split('_')[0]
            feat_data = np.load(os.path.join(cwd_path, "data/npy/feat_data", file))
            len_data = np.load(os.path.join(cwd_path, "data/npy/len_data", name + "_len.npy"))
            protobuf_vec = InProtobufVector()
            mxpi_tensor_package_list = MxpiDataType.MxpiTensorPackageList()
            tensor_package_vec = mxpi_tensor_package_list.tensorPackageVec.add()

            # add feature data #begin
            tensorVec = tensor_package_vec.tensorVec.add()
            tensorVec.memType = 1
            tensorVec.deviceId = 0
            # Compute the number of bytes of feature data.
            tensorVec.tensorDataSize = int(
                feat_data.shape[1]*feat_data.shape[2]*4)
            tensorVec.tensorDataType = 0  # float32
            for i in feat_data.shape:
                tensorVec.tensorShape.append(i)
            tensorVec.dataStr = feat_data.tobytes()   
            # add feature data #end

            # add length data #begin
            tensorVec2 = tensor_package_vec.tensorVec.add()
            tensorVec2.memType = 1
            tensorVec2.deviceId = 0
            # Compute the number of bytes of length data.
            # int(4)  4: btyes of int32
            tensorVec2.tensorDataSize = int(4)
            tensorVec2.tensorDataType = 3  # int32
            for i in len_data.shape:
                tensorVec2.tensorShape.append(i)
            tensorVec2.dataStr = len_data.tobytes()
            # add length data #end

            protobuf = MxProtobufIn()
            protobuf.key = TENSOR_KEY
            protobuf.type = b'MxTools.MxpiTensorPackageList'
            protobuf.protobuf = mxpi_tensor_package_list.SerializeToString()
            protobuf_vec.push_back(protobuf)
            start = time.time()
            unique_id = streamManager.SendProtobuf(
                STREAM_NAME, inPluginId, protobuf_vec)
            if unique_id < 0:
                print("Failed to send data to stream.")
                exit()

            key_vec = StringVector()
            key_vec.push_back(b'mxpi_tensorinfer1')
            # get inference result
            infer_result = streamManager.GetProtobuf(
                STREAM_NAME, inPluginId, key_vec)
            if infer_result.size() == 0:
                print("infer_result is null")
                exit()
            if infer_result[0].errorCode != 0:
                print("GetProtobuf error. errorCode=%d" % (
                    infer_result[0].errorCode))
                exit()

            result = MxpiDataType.MxpiTensorPackageList()
            result.ParseFromString(infer_result[0].messageBuf)
            # converts the inference result to a NumPy array
            ids = np.frombuffer(
                result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.int32)
            end = time.time()
            cost_all += (end - start)
            # decode
            lm_tokens_path = os.path.join(cwd_path, "data/lm_tokens.txt")
            text_featurizer = TextFeaturizer(lm_tokens_path)
            text = text_featurizer.deocde_without_start_end(ids)
            # convert list to string and print recognition result
            text_list.append(''.join(name)+' '+''.join(text)+'\n')
            wav_count += 1

    with os.fdopen(os.open(cwd_path + "/data/prediction.txt", FLAGS, MODES), 'w') as fout:
        for item in text_list:       
            fout.writelines(item)

    import re
    f0 = open(cwd_path + "/data/prediction.txt", 'r+')
    h0 = f0.readlines()
    for file in dirs:            
        if file.endswith(".txt"):
            name = file.split('.')[0]
            file1 = name + '.txt'
            f = open(cwd_path + "/data/sample/" + file1)
            r = f.readline()
            len_predi = len(h0)
            for k in range(len_predi):
                match = re.search(name, h0[k])
                if match is not None:
                    print ("当前语音文件名：", file1 + '\n')
                    print ("正确文件内容: ", r)
                    h1 = h0[k].split(' ')[1]
                    r = [x for x in r]
                    h = [x for x in h1]
                    print ("推理结果：", h1)
                    cer_value, (s, d, i) = levenshtein(h, r)
                    wer_s += s
                    wer_i += i
                    wer_d += d
                    wer_n += len(h)         

    # if data is numpy file       
    print('替换:{0},插入:{1},删除:{2},总字数:{3},字错率（CER）:{4}'.format(
        wer_s, wer_i, wer_d, wer_n, (wer_s + wer_i + wer_d) / wer_n))           
    print ("{}条语音的推理时长是：{}秒".format(wav_count, cost_all))