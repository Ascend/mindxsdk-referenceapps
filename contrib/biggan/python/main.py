#!/usr/bin/env python
# coding=utf-8

"""
Copyright(C) Huawei Technologies Co.,Ltd. 2012-2022 All rights reserved.

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

import cv2


import MxpiDataType_pb2 as MxpiDataType

from StreamManagerApi import StreamManagerApi, MxProtobufIn, InProtobufVector, StringVector




def preprocess(l_path, n_path, num, count):


    label_files = os.listdir(l_path)
    noise_files = os.listdir(n_path)
    
    label_files.sort(key=lambda x:int(x.split('input_')[1].split('.bin')[0]))
    noise_files.sort(key=lambda x:int(x.split('input_')[1].split('.bin')[0]))
    
    print(label_files[count], noise_files[count])
    # gen np array
    label = np.fromfile(os.path.join(l_path, label_files[count]), dtype = np.float32, count = -1)
    label.shape = 1, 5, 148
    noise = np.fromfile(os.path.join(n_path, noise_files[count]), dtype = np.float32, count = -1)
    noise.shape = 1, 1, 20

    print("prepare_bin success")
    
    # gen tensor data
    mxpi_tensor_pack_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package_vec = mxpi_tensor_pack_list.tensorPackageVec.add()
    
    print("preprocess tensor success")

    

    # add noise data
    tensorvec_noise = tensor_package_vec.tensorVec.add()
    tensorvec_noise.memType = 1
    tensorvec_noise.deviceId = 0
    # bsΪ1�� 1*20Ϊ��������ߴ磬4ָ��float32����
    tensorvec_noise.tensorDataSize = int(1*1*20*4)
    tensorvec_noise.tensorDataType = 0
    for i in noise.shape:
        tensorvec_noise.tensorShape.append(i)
    tensorvec_noise.dataStr = noise.tobytes()
    
    # add label data
    tensorvec_label = tensor_package_vec.tensorVec.add()
    tensorvec_label.memType = 1
    tensorvec_label.deviceId = 0
    # bsΪ1�� 5*148Ϊ��ǩ����ߴ磬4ָ��float32����
    tensorvec_label.tensorDataSize = int(1*5*148*4) 
    tensorvec_label.tensorDataType = 0


    for i in label.shape:
        tensorvec_label.tensorShape.append(i)
    tensorvec_label.dataStr = label.tobytes()

    return mxpi_tensor_pack_list


if __name__ == '__main__':


    # init stream manager
    stream_manager = StreamManagerApi()
    ret = stream_manager.InitManager()
    if ret != 0:
        print(" Init Stream manager failure , ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./biggan.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Create Stream failure, ret=%s" % str(ret))
        exit()


    # set stream name and device
    STREAM_NAME = b'biggan'
    IN_PLUGIN_ID = 0
    #  Position of label in data set
    LABEL_PATH = '../prep_label_bs1'
    #  Location of noise in data set
    NOISE_PATH = '../prep_noise_bs1'
    #  Position of output image
    OUT_PATH = '../result'
    OUT_PATH_BIN = "../result_bin"
    #  Dataset range
    NUM = 10000    
    #  Number of generated image
    COUNT = 0

    for COUNT in range(NUM):
        print(COUNT)
        
        
        if COUNT >= NUM:
            print("set COUNT again, should be smaller than NUM.")
            exit()

        if not os.path.isdir(LABEL_PATH):
            print("LabelPath does not exit.")
            exit()
        
        if not os.path.isdir(NOISE_PATH):
            print("NoisePath does not exit.")
            exit()
            
        tensor_pack_list = preprocess(LABEL_PATH, NOISE_PATH, NUM, COUNT)
        
        
        # send data to stream
        proto_buffer_in = MxProtobufIn()
        proto_buffer_in.key = b'appsrc0'
        proto_buffer_in.type = b'MxTools.MxpiTensorPackageList'
        proto_buffer_in.protobuf = tensor_pack_list.SerializeToString()

        proto_buffer_vec = InProtobufVector()
        proto_buffer_vec.push_back(proto_buffer_in)

        ret = stream_manager.SendProtobuf(STREAM_NAME, IN_PLUGIN_ID, proto_buffer_vec)
        if ret < 0:
            print("Send data failure., ret=%s" % str(ret))
            exit()
        
        # get inference result
        keys = [b"mxpi_tensorinfer0"]
        key_vec = StringVector()
        for key in keys:
            key_vec.push_back(key)

        
        result_raw = stream_manager.GetResult(STREAM_NAME, b'appsink0', key_vec)
        SIZE_INFER_RAW = result_raw.metadataVec.size()
        print("result.metadata size: ", SIZE_INFER_RAW)
        result_metadata = result_raw.metadataVec[0]
    
        if result_metadata.errorCode != 0:
            print("GetResult error. errorCode=%d , errMsg=%s" % (
                result_metadata.errorCode, result_metadata.errMsg))
            exit()

    
        # convert result
        result_tensor_pack_list = MxpiDataType.MxpiTensorPackageList()
        result_tensor_pack_list.ParseFromString(result_metadata.serializedMetadata)

        print("tensorPackageVec size=%d, tensorPackageVec[0].tensorVec size=%d" % ( 
            len(result_tensor_pack_list.tensorPackageVec), len(result_tensor_pack_list.tensorPackageVec[0].tensorVec)))
        result_tensor = result_tensor_pack_list.tensorPackageVec[0].tensorVec[0]
            
        if not os.path.isdir(OUT_PATH):
            os.makedirs(OUT_PATH)
        
        if not os.path.isdir(OUT_PATH_BIN):
            os.makedirs(OUT_PATH_BIN)

        img = np.frombuffer(result_tensor.dataStr
            , dtype = np.float32)
        print("raw output shape:", result_tensor.tensorShape)  
    
        img.resize(3, 128, 128)
    
        print("img.shape: ", img.shape)
        # 3, 128, 128
        print(img)
        # to bin
        j = "%04d" % (COUNT)
        filepath = OUT_PATH_BIN + "/"+str(j)+"_result.bin"
        img.tofile(filepath)

        image =  img.copy()
        img_int = img.copy()
        # The mean parameter set in dataLoader
        mean = [0.485, 0.456, 0.406] 
        # Std parameter set in dataLoader
        std = [0.229, 0.224, 0.225]  

        #Denormalization
        mean_len = range(len(mean))
        for a in mean_len: 
            image[a] = img[a] * std[a] + mean[a]
        image = image * 255 
        image = np.transpose(image, (1, 2, 0))
        img_int = image.astype(np.uint8)

        print(img_int)
    
        img_bgr = cv2.cvtColor(img_int, cv2.COLOR_RGB2BGR)
        cv2.imwrite(OUT_PATH + "/"+str(j)+"_result.jpg", img_bgr)





