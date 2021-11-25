"""
Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.

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

import numpy as np 
import os 
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
import MxpiDataType_pb2 as MxpiDataType
import cv2

def calc_mad(output, mask):
    """
        Compute the accuracy for each picture in the test set.
        Args:
            the pictures in the test set, 
            the picture mask in the mask set
        Returns: 
            the test accuracy mad
    """
    mad = np.mean(np.abs(mask - output))
    return mad


if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    pipeline_path = "./pipeline/MMNET.pipeline"
    if os.path.exists(pipeline_path) != 1:
        print("Failed to get the pipeline correctly. Please check it!")
        exit()
    with open(pipeline_path, 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    dataInput = MxDataInput()
    streamName = b'mmnet'
    inPluginId = 0
    keyVec = StringVector()
    keyVec.push_back(b"mxpi_tensorinfer0")

    filepath = "./test/"
    if os.path.isdir(filepath) != 1:
        print("Failed to get the input folder. Please check it!")
        exit()
    gt_dir = './mask'
    if os.path.isdir(gt_dir) != 1:
        print("Failed to get the mask input folder. Please check it!")
        exit()
    
    output_shape = [1, 256, 256, 2]
    gt_shape = [1, 256, 256, 1]

    names_input = sorted(os.listdir(filepath))
    names_gt = sorted(os.listdir(gt_dir))

    total_mad = 0

    for i, j in zip(names_input, names_gt):
        filename_input = os.path.join(filepath, i)
        if os.path.exists(filename_input) != 1:
            print("Failed to get the input picture. Please check it!")
            exit()
        filename_gt = os.path.join(gt_dir, j)
        if os.path.exists(filename_gt) != 1:
            print("Failed to get the input picture mask. Please check it!")
            exit()
        with open(filename_input, 'rb') as f:
            dataInput.data = f.read()

        uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)
        if uniqueId < 0:
            print("Failed to send data to stream.")
            exit()
        infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)
        if infer_result.size() == 0:
            print("infer_result is null")
            exit()
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
                infer_result[0].errorCode, infer_result[0].data.decode()))
            exit()
        tensorList = MxpiDataType.MxpiTensorPackageList()
        tensorList.ParseFromString(infer_result[0].messageBuf)
        prediction = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype = np.float32)
        prediction = prediction.reshape(output_shape)[:, :, :, -1:]
        
        gt = cv2.imread(filename_gt, cv2.IMREAD_GRAYSCALE)
        gt = np.array(cv2.resize(gt, (256, 256)), dtype=np.float32).reshape(gt_shape) / 255.0
        total_mad += calc_mad(gt, prediction)
    print(len(names_gt))
    print(total_mad / len(names_gt))

