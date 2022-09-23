# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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

import sys
import math
import datetime
import pandas as pd
import numpy as np
from sklearn import preprocessing
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, InProtobufVector, MxProtobufIn, StringVector


def send_source_data(appsrc_id, tensor, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.
    """
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    array_bytes = tensor.tobytes()
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for i in tensor.shape:
        tensor_vec.tensorShape.append(i)
    tensor_vec.dataStr = array_bytes
    tensor_vec.tensorDataSize = len(array_bytes)
    key = "appsrc{}".format(appsrc_id).encode('utf-8')
    protobuf_vec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensor_package_list.SerializeToString()
    protobuf_vec.push_back(protobuf)

    ret = stream_manager.SendProtobuf(stream_name, appsrc_id, protobuf_vec)
    return ret


def load_data(dir_name, n_his):
    data_frame = pd.read_csv(dir_name, header=None)
    data = data_frame

    zscore.fit(data_frame[:])
    data = zscore.transform(data)

    n_route = data.shape[1]
    dayslot = len(data)
    n_slot = dayslot - n_his

    x = np.zeros([n_slot, 1, n_his, n_route], np.float32)

    for i in range(n_slot):
        first = i
        last = i + n_his
        x[i, :, :, :] = data[first: last].reshape(1, n_his, n_route)
    return x, n_slot


def get_infer_result(stream_name, inplugin_id):
    start_time = datetime.datetime.now()

    key_vec = StringVector()
    key_vec.push_back(b'mxpi_tensorinfer0')
    pipeline_result = stream_manager_api.GetProtobuf(stream_name, inplugin_id, key_vec)

    end_time = datetime.datetime.now()
    print('sdk run time: {}'.format((end_time - start_time).microseconds))

    if pipeline_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" % (pipeline_result[0].errorCode))
        sys.exit()
    # get infer result
    out_result = MxpiDataType.MxpiTensorPackageList()
    out_result.ParseFromString(pipeline_result[0].messageBuf)

    return out_result


if __name__ == '__main__':
    if len(sys.argv) == 3:
        dir_name = sys.argv[1]
        res_dir_name = sys.argv[2]
    else:
        print("ERROR, please enter again.")
        exit(1)
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    # create streams by pipeline config file
    with open("./pipeline/stgcn.pipeline", 'rb') as f:
        pipeline_string = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_string)

    # Construct the input of the stream
    n_his = 12
    zscore = preprocessing.StandardScaler()
    # 读数据集
    x, n_slot = load_data(dir_name, n_his)
    predictions = []
    stream_name = b'im_stgcn'
    #start infer
    for i in range(n_slot):
        inplugin_id = 0
        tensor = np.expand_dims(x[i], axis=0)
        uniqueid = send_source_data(0, tensor, stream_name, stream_manager_api)
        if uniqueid < 0:
            print("UniqueID ERROR")
            sys.exit()

        # Obtain the inference result by specifying stream_name and uniqueId.
        out_result = get_infer_result(stream_name, inplugin_id)

        # convert the inference result to Numpy array
        res = np.frombuffer(out_result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        predictions.append(zscore.inverse_transform(np.expand_dims(res, axis=0)).reshape(-1))

    np.savetxt(res_dir_name + 'predcitions.txt', np.array(predictions))

    # destroy streams
    stream_manager_api.DestroyAllStreams()
