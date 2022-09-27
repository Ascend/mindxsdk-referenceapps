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


def send_data(appsrc_id, tensor, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.
    """
    tpackage_list = MxpiDataType.MxpiTensorPackageList()
    array_bytes = tensor.tobytes()
    tpackage = tpackage_list.tensorPackageVec.add()
    tvec = tpackage.tensorVec.add()
    tvec.deviceId = 0
    tvec.memType = 0
    for i in tensor.shape:
        tvec.tensorShape.append(i)
    tvec.dataStr = array_bytes
    tvec.tensorDataSize = len(array_bytes)
    key = "appsrc{}".format(appsrc_id).encode('utf-8')
    pf_vec = InProtobufVector()
    pf = MxProtobufIn()
    pf.key = key
    pf.type = b'MxTools.MxpiTensorPackageList'
    pf.protobuf = tpackage_list.SerializeToString()
    pf_vec.push_back(pf)

    tmp = stream_manager.SendProtobuf(stream_name, appsrc_id, pf_vec)
    return tmp


def load_data(dir_name, n_his, n_pred):
    data_frame = pd.read_csv(dir_name, header=None)
    data_col = data_frame.shape[0]
    if data_frame.shape[1] != 156:
        print("The data set format does not meet the requirements!")
        sys.exit()
    split_rate = 0.15
    val_len = int(math.floor(data_col * split_rate))
    test_len = int(math.floor(data_col * split_rate))
    train_len = int(data_col - val_len - test_len)
    data = data_frame[train_len + val_len:]

    zscore.fit(data_frame[: train_len + val_len])
    data = zscore.transform(data)

    n_route = data.shape[1]
    dayslot = len(data)
    n_slot = dayslot - n_his - n_pred

    x = np.zeros([n_slot, 1, n_his, n_route], np.float32)
    y = np.zeros([n_slot, n_route], np.float32)
    a = 0
    while a < n_slot:
        first = a
        last = a + n_his
        x[a, :, :, :] = data[first: last].reshape(1, n_his, n_route)
        y[a] = data[last + n_pred - 1]
        a += 1

    return x, y, n_slot


def get_infer_result(stream_name, inplugin_id, stream_manager_api):
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


def test(x, y):
    mae, mse = [], []
    for pre, label in zip(np.array(x), np.array(y)):
        if np.isnan(label).sum() > 0:
            continue
        else:
            d = np.abs(pre - label)
            mae += d.tolist()
            mse += (d ** 2).tolist()

    mae_result = np.array(mae).mean()
    rmse_result = np.sqrt(np.array(mse).mean())
    print(f'MAE {mae_result:.2f} | RMSE {rmse_result:.2f} ')


if __name__ == '__main__':
    if len(sys.argv) == 4:
        dirname = sys.argv[1]
        resdirname = sys.argv[2]
        npred = int(sys.argv[3])
    else:
        print("ERROR, please enter again.")
        exit(1)

    start_time_all = datetime.datetime.now()
    streaminput_manager_api = StreamManagerApi()
    ret = streaminput_manager_api.InitManager()
    # create streams by pipeline config file
    with open("./pipeline/stgcn.pipeline", 'rb') as f:
        pipeline_string = f.read()
    ret = streaminput_manager_api.CreateMultipleStreams(pipeline_string)

    # Construct the input of the stream
    NHIS = 12
    zscore = preprocessing.StandardScaler()
    # 读数据集
    x_pre, y_lab, nslot = load_data(dirname, NHIS, npred)
    labels = []
    predictions = []
    #start infer
    k = 0
    while k < nslot:
        tensor_input = np.expand_dims(x_pre[k], axis=0)
        uniqueid = send_data(0, tensor_input, b'im_stgcn', streaminput_manager_api)
        if uniqueid < 0:
            print("UniqueID ERROR")
            sys.exit()

        # Obtain the inference result by specifying stream_name and uniqueId.
        result = get_infer_result(b'im_stgcn', 0, streaminput_manager_api)

        # convert the inference result to Numpy array
        res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        labels.append(zscore.inverse_transform(np.expand_dims(y_lab[k], axis=0)).reshape(-1))
        predictions.append(zscore.inverse_transform(np.expand_dims(res, axis=0)).reshape(-1))
        k += 1

    np.savetxt(resdirname+'labels.txt', np.array(labels))
    np.savetxt(resdirname+'predcitions.txt', np.array(predictions))

    test(predictions, labels)

    end_time_all = datetime.datetime.now()
    print('total time: {}'.format((end_time_all - start_time_all).microseconds))
    # destroy streams
    streaminput_manager_api.DestroyAllStreams()
