# coding=utf-8

"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import datetime
import json
import os
import sys
import math
import numpy as np

from StreamManagerApi import StreamManagerApi
from StreamManagerApi import MxDataInput


def gui_yi_hua(predicted):
    predicted = np.array(predicted)
    add_val = 0
    min_val = np.min(predicted)
    if min_val < 0 :
        add_val = -math.floor(min_val)
    predicted = predicted + add_val
    print("add_val:{}, min_val:{} ".format(add_val, min_val))
    dists = np.linalg.norm(predicted, axis=1, keepdims=1)
    out = np.where(np.isclose(dists, 0), 0, predicted / dists)
    return out


def multiclass_logloss(actual, predicted, eps=1e-15):
    """Logarithmic Loss  Metric
    :param actual: 包含actual target classes的数组
    :param predicted: 分类预测结果矩阵, 每个类别都有一个概率
    """
    # Convert 'actual' to a binary array if it's not already:
    print(predicted.shape)
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[-1]))
        for i, value in enumerate(actual):
            actual2[i, value] = 1
        actual = actual2
        

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    logloss = -1.0 / rows * vsota
    print("LogLoss:",logloss)
    return val

if __name__ == '__main__':
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./pipeline/dirver-detection-img.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    data_input = MxDataInput()
    dir_name = sys.argv[1]
    file_list = os.listdir(dir_name)
    
    result = []
    actual_label = []
    
    for file_label in file_list:
        file_label_path = os.path.join(dir_name, file_label)
        for file_name in os.listdir(file_label_path):
            
            file_path = os.path.join(file_label_path, file_name)
            if not (file_name.lower().endswith(".jpg") or file_name.lower().endswith(".jpeg")):
                continue

            with open(file_path, 'rb') as f:
                data_input.data = f.read()
        
            empty_data = []
            stream_name = b'im_resnet50'
            in_plugin_id = 0
            unique_id = stream_manager_api.SendData(stream_name, in_plugin_id, data_input)
            if unique_id < 0:
                print("Failed to send data to stream.")
                exit()
            # Obtain the inference result by specifying streamName and uniqueId.
            start_time = datetime.datetime.now()
            infer_result = stream_manager_api.GetResult(stream_name, unique_id)
            end_time = datetime.datetime.now()
            print('sdk run time: {}'.format((end_time - start_time).microseconds))
            if infer_result.errorCode != 0:
                print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                    infer_result.errorCode, infer_result.data.decode()))
                exit()
            # print the infer result
            infer_res = infer_result.data.decode()
            print("process img: {}, infer result: {}".format(file_name, infer_res))
        
            cls_id = infer_res.split(",")[0].split("{[")[-1]
            res = eval(infer_res)
            tmp = [0,0,0,0,0,0,0,0,0,0]
            for val in res["MxpiClass"]:
                idx = val["classId"]
                tmp[idx] = val["confidence"] 
            
            result.append(tmp)
            actual_label.append(int(file_label.split("c")[-1]))
            
    multiclass_logloss(np.array(actual_label),  gui_yi_hua(result))
    # destroy streams
    stream_manager_api.DestroyAllStreams()
