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
import os
import time

import utils
import numpy as np
from preprocessing import ExtractLogmel
import MxpiDataType_pb2 as MxpiDataType

from StreamManagerApi import StreamManagerApi, InProtobufVector, MxProtobufIn, StringVector, MxDataInput

if __name__ == "__main__":
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
    extract_logmel = ExtractLogmel(padded_type="copy")
    # Path for storing test data
    wav_dir = "/home/tianyinghui/Data/data_aishell/wav/dev"
    speakers = os.listdir(wav_dir)
    speakers.sort()
    all_wav_num = 0
    # each speaker choice one wav to register
    start = time.time()
    for idx, speaker in enumerate(speakers):
        # get wav path
        all_wav_paths = utils.get_all_type_paths(os.path.join(wav_dir, speaker), ".wav")
        all_wav_paths.sort()
        all_wav_num += len(all_wav_paths)
        for index, wav_path in enumerate(all_wav_paths):
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
            res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
            # save speakers' embedding
            if index < 1:
                # embedding for register
                enroll_embedding_save_dir = "/home/tianyinghui/Data/spk_data/embeddings/om_dev_speaker_enroll"
                os.makedirs(enroll_embedding_save_dir, exist_ok=True)
                np.save(os.path.join(enroll_embedding_save_dir, speaker + ".npy"), res)
            else:
                # embedding for recognition
                eval_embedding_save_dir = "/home/tianyinghui/Data/spk_data/embeddings/om_dev_speaker_eval/{}"\
                    .format(speaker)
                os.makedirs(eval_embedding_save_dir, exist_ok=True)
                basename = os.path.basename(wav_path).replace(".wav", ".npz")
                np.savez(os.path.join(eval_embedding_save_dir, basename),
                         embedding=res,
                         speaker=speaker)
    print("Embedding extraction of the speaker is complete!")
    end = time.time()
    print("{} samples' infer time:{}".format(all_wav_num, end-start))
    # get trials files
    utils.get_trials("/home/tianyinghui/Data/spk_data/embeddings/om_dev_speaker_enroll",
                     "/home/tianyinghui/Data/spk_data/embeddings/om_dev_speaker_eval",
                     "/home/tianyinghui/Data/spk_data/embeddings/dev_eval_trails.txt")
    print("get trials!")
    # compute eer
    EER = utils.cal_eer("/home/tianyinghui/Data/spk_data/embeddings/dev_eval_trails.txt")












