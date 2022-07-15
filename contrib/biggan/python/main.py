#!/usr/bin/env python
# coding=utf-8

"""
Copyright(C) Huawei Technologies Co.,Ltd. 2012-2021 All rights reserved.

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
# USE LF FORMAT!

import os
import time
import numpy as np
import torch

from torchvision.utils import save_image

import MxpiDataType_pb2 as MxpiDataType

from StreamManagerApi import StreamManagerApi, MxProtobufIn, InProtobufVector, StringVector




def preprocess(label_path, noise_path, num, count):


    label_files = os.listdir(label_path)
    noise_files = os.listdir(noise_path)
    
    label_files.sort(key=lambda x:int(x.split('input_')[1].split('.bin')[0]))
    noise_files.sort(key=lambda x:int(x.split('input_')[1].split('.bin')[0]))
    
    print(label_files[count], noise_files[count])
    
    label = np.fromfile(os.path.join(label_path, label_files[count]), dtype = np.float32, count = -1)
    label.shape = 1, 5, 148
    noise = np.fromfile(os.path.join(noise_path, noise_files[count]), dtype = np.float32, count = -1)
    noise.shape = 1, 1, 20
    # gen np array


    
    print("prepare_bin success")
    
    # gen tensor data
    mxpi_tensor_pack_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package_vec = mxpi_tensor_pack_list.tensorPackageVec.add()
    
    print("preprocess tensor success")

    

    # add noise data
    tensorvec_noise = tensor_package_vec.tensorVec.add()
    tensorvec_noise.memType = 1
    tensorvec_noise.deviceId = 0
    tensorvec_noise.tensorDataSize = int(1*1*20*4)
     # H*W*C*(float32)
    tensorvec_noise.tensorDataType = 0
     # float32
    for i in noise.shape:
        tensorvec_noise.tensorShape.append(i)
    tensorvec_noise.dataStr = noise.tobytes()
    
    # add label data
    tensorvec_label = tensor_package_vec.tensorVec.add()
    tensorvec_label.memType = 1
    tensorvec_label.deviceId = 0
    tensorvec_label.tensorDataSize = int(1*5*148*4) 
    tensorvec_label.tensorDataType = 0


    for i in label.shape:
        tensorvec_label.tensorShape.append(i)
    tensorvec_label.dataStr = label.tobytes()

    return mxpi_tensor_pack_list


if __name__ == '__main__':
    # set stream name and device
    streamName = b'biggan'
    IN_PLUGIN_ID = 0
  
    label_path = '../prep_label_bs1'
    noise_path = '../prep_noise_bs1'
    out_path = '../result'
    NUM = 1000    
    #数据集范围
    COUNT = 11    
    #需要生成的编号
    tensor_pack_list = preprocess(label_path, noise_path, NUM, COUNT)

    # init stream manager
    stream_manager = StreamManagerApi()
    ret = stream_manager.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./biggan.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()



    # send data to stream
    protobuf_in = MxProtobufIn()
    protobuf_in.key = b'appsrc0'
    protobuf_in.type = b'MxTools.MxpiTensorPackageList'
    protobuf_in.protobuf = tensor_pack_list.SerializeToString()

    protobuf_vec = InProtobufVector()
    protobuf_vec.push_back(protobuf_in)
    
    time_start = time.time()
    unique_id = stream_manager.SendProtobuf(stream_name, in_plugin_id, protobuf_vec)
    if unique_id < 0:
        print("Failed to send data to stream.")
        exit()

    # get inference result
    keys = [b"mxpi_tensorinfer0"]
    key_vec = StringVector()
    for key in keys:
        key_vec.push_back(key)

    infer_raw = stream_manager.GetResult(streamName, b'appsink0', key_vec)
    print("result.metadata size: ", infer_raw.metadataVec.size())
    infer_result = infer_raw.metadataVec[0]
    


    if infer_result.errorCode != 0:
        print("GetResult error. errorCode=%d , errMsg=%s" % (
            infer_result.errorCode, infer_result.errMsg))
        exit()
    time_end = time.time()
    print('Time cost = %fms' % ((time_end - time_start) * 1000))


    # convert result
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result.serializedMetadata)
    print("tensorPackageVec size=%d, tensorPackageVec[0].tensorVec size=%d" % ( 
        len(result.tensorPackageVec), len(result.tensorPackageVec[0].tensorVec)))
        
    if not os.path.isdir(out_path):
        os.makedirs(out_path)


    img = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr
        , dtype = np.float32)
    print("raw output shape:", result.tensorPackageVec[0].tensorVec[0].tensorShape)   #1,3,128,128
 
    shape = (-1,3,128,128)
    img = torch.from_numpy(img.copy())
    img = img.view(shape)
    bs=1
    bs, _, _, _ = img.shape
    baseName = os.path.join(str(COUNT) + "_result")
    target_path = os.path.join(out_path, baseName + ".jpg")
    save_image(img[0], normalize = True, nrow = 1, fp = target_path)
    
    