# /*
#  * Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  * you may not use this file except in compliance with the License.
#  * You may obtain a copy of the License at
#  *
#  * http://www.apache.org/licenses/LICENSE-2.0
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS,
#  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  * See the License for the specific language governing permissions and
#  * limitations under the License.
#  */

import os
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import *

from pre_process import make_model_input
from post_process import TextFeaturizer


if __name__ == "__main__":

    cwd_path = os.getcwd()
    pipeline_path = os.path.join(cwd_path, "pipeline/am_lm.pipeline")

    stream_name = b'speech_recognition'
    tensor_key = b'appsrc0'
    in_plugin_id = 0

    stream_manager = StreamManagerApi()
    ret = stream_manager.InitManager()
    if ret != 0:
        print("Failed to init stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(pipeline_path, 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    # if the data are raw file
    wav_file_path = os.path.join(cwd_path, "data/BAC009S0009W0133.wav")
    feat_data, len_data = make_model_input([wav_file_path])
    # feat_data = np.load(os.path.join(cwd_path, "data/feat_data.npy"))
    # len_data = np.load(os.path.join(cwd_path, "data/len_data.npy"))

    protobuf_vec = InProtobufVector()
    mxpi_tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package_vec = mxpi_tensor_package_list.tensorPackageVec.add()

    # add feature data begin
    tensorVec = tensor_package_vec.tensorVec.add()
    tensorVec.memType = 1
    tensorVec.deviceId = 0
    tensorVec.tensorDataSize = int(1001 * 80 * 4)
    tensorVec.tensorDataType = 0
    tensorShape = [1, 1001, 80, 1]
    for i in range(4):
        tensorVec.tensorShape.append(tensorShape[i])
    tensorVec.dataStr = feat_data.tobytes()
    # add feature data end

    # add length data begin
    tensorVec2 = tensor_package_vec.tensorVec.add()
    tensorVec2.memType = 1
    tensorVec2.deviceId = 0
    tensorVec2.tensorDataSize = int(4)
    tensorVec2.tensorDataType = 3
    tensorShape2 = [1, 1]
    for i in range(2):
        tensorVec2.tensorShape.append(tensorShape2[i])
    tensorVec2.dataStr = len_data.tobytes()
    # add length data end

    protobuf = MxProtobufIn()
    protobuf.key = tensor_key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = mxpi_tensor_package_list.SerializeToString()
    protobuf_vec.push_back(protobuf)

    unique_id = stream_manager.SendProtobuf(
        stream_name, in_plugin_id, protobuf_vec)
    # uniqueId = stream_manager.SendDataWithUniqueId(stream_name, in_plugin_id, dataInput)
    if unique_id < 0:
        print("Failed to send data to stream.")
        exit()

    key_vec = StringVector()
    key_vec.push_back(b'mxpi_tensorinfer1')
    infer_result = stream_manager.GetProtobuf(
        stream_name, in_plugin_id, key_vec)
    if infer_result.size() == 0:
        print("infer_result is null")
        exit()
    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" % (
            infer_result[0].errorCode))
        exit()

    # print some inference information
    print("infer_result size: ", len(infer_result))
    # print the infer result
    print("GetProtobuf errorCode=%d" % (infer_result[0].errorCode))
    print("key:" + str(infer_result[0].messageName))
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)
    # print(result.tensorPackageVec[0].tensorVec[0].dataStr)
    print("result.tensorPackageVec size: ", len(result.tensorPackageVec))
    print("result.tensorPackageVec[0].tensorVec size: ", len(
        result.tensorPackageVec[0].tensorVec))

    # converts the inference result to a NumPy array
    ids = np.frombuffer(
        result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.int32)

    # decode
    lm_tokens_path = os.path.join(cwd_path, "data/lm_tokens.txt")
    text_featurizer = TextFeaturizer(lm_tokens_path)
    text = text_featurizer.deocde_without_start_end(ids)
    print("The recognition result: ", text)
