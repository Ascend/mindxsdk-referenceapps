#!/usr/bin/env python
# coding=utf-8

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
import cv2
import numpy as np
import torch
import os
from PIL import Image
import argparse
import pickle as pickle
from PIL import Image
import torchvision.transforms as transforms
import random
from evaluate import attribute_evaluate_lidw
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector, InProtobufVector, MxProtobufIn

if __name__ == '__main__':

    # load data
    image = []
    label = []
    with open("dataset/peta_dataset.pkl", 'rb') as data_file:
        dataset = pickle.load(data_file)
    with open("dataset/peta_partition.pkl", 'rb') as data_file:
        partition = pickle.load(data_file)
    for idx in partition['test'][0]:
        image.append(dataset['test_image'][idx])
        label_tmp = np.array(dataset['att'][idx])[dataset['selected_attribute']].tolist()
        label.append(label_tmp)

    streamName = b"detection"
    inPluginId = 0

    streamManagerApi = StreamManagerApi()
    # init stream manager
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    pipeline_path = b"pipeline/test_only_deepmar.pipeline"
    tensor_key = b'appsrc0'
    ret = streamManagerApi.CreateMultipleStreamsFromFile(pipeline_path)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    start_epoch = 0
    uniqueIds = []
    valid_idx = []

    # Filter the test pictures in the dataset
    filePath = 'dataset/image_jpg'
    jpg_name = os.listdir(filePath)
    valid_img = []
    invalid_img = []
    # label_index = []
    label_selected = []
    for name in image:
        if name[0:5] + '.jpg' not in jpg_name:
            continue
        im = Image.open('dataset/image_jpg/' + name[0:5] + '.jpg')
        height, width = im.size
        if width == 160 and height == 80:
            valid_img.append(name)
            label_selected.append(label[image.index(name)])


    valid_img_selected = valid_img
    label_selected_ = label_selected
    j = 0
    valid_label = []

    keyVec = StringVector()
    keyVec.push_back(b"mxpi_tensorinfer0")

    # Normalize and standardize the test image
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize, ])

    # Collect model inferencing results
    for i in range(0, len(valid_img_selected)):
        img_path = "dataset/image_jpg/" + valid_img_selected[i][0:5] + '.jpg'
        dataInput = MxDataInput()
        img = Image.open(img_path)
        img_trans = test_transform(img)
        img = np.array(img_trans)
        image = img.reshape(1, 3, 224, 224)

        image = image.astype(np.float32)
        protobuf_vec = InProtobufVector()
        mxpi_tensor_package_list = MxpiDataType.MxpiTensorPackageList()
        tensor_package_vec = mxpi_tensor_package_list.tensorPackageVec.add()
        tensorVec = tensor_package_vec.tensorVec.add()
        tensorVec.memType = 1
        tensorVec.deviceId = 0
        tensorVec.tensorDataSize = int(
            img.shape[1] * img.shape[2])
        tensorVec.tensorDataType = 0
        for m in image.shape:
            tensorVec.tensorShape.append(m)
        tensorVec.dataStr = image.tobytes()

        protobuf = MxProtobufIn()
        protobuf.key = tensor_key
        protobuf.type = b'MxTools.MxpiTensorPackageList'
        protobuf.protobuf = mxpi_tensor_package_list.SerializeToString()
        protobuf_vec.push_back(protobuf)
        uniqueId = streamManagerApi.SendProtobuf(
            streamName, 0, protobuf_vec)

        if uniqueId < 0:
            print("Failed to send data to stream.")
            exit()
        uniqueIds.append(uniqueId)

        # Receive results
        infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)

        if len(infer_result) != 1:
            invalid_img.append(valid_img[i])
            continue

        if len(infer_result) == 1:
            valid_idx.append(valid_img[i])
            tensorList = MxpiDataType.MxpiTensorPackageList()
            tensorList.ParseFromString(infer_result[0].messageBuf)
            feat_tmp = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
            print(feat_tmp)
            if i == 0:
                feat = np.zeros((4500, 35))
            feat[j:j + 1, :] = feat_tmp.reshape((1, -1))
            j = j + 1
            valid_label.append(label_selected_[i])
    # the collected inferencing results for tagging
    row_zero = int(np.where(~feat.any(axis=1))[0][0])
    pt_result = feat[0:row_zero - 1, :]
    gt_result = np.zeros(pt_result.shape)
    for idx, label_meta in enumerate(valid_label[0:row_zero - 1]):
        gt_result[idx, :] = label_meta
    pt_result[pt_result >= 0] = 1
    pt_result[pt_result < 0] = 0

    # Calculation accuracy results
    result = attribute_evaluate_lidw(gt_result, pt_result)
    print('Label-based evaluation: \n mA: %.4f' % (np.mean(result['label_acc'])))
    print('Instance-based evaluation: \n Acc: %.4f, Prec: %.4f, Rec: %.4f, F1: %.4f' \
          % (result['instance_acc'], result['instance_precision'], result['instance_recall'], result['instance_F1']))

    # destroy streams
    streamManagerApi.DestroyAllStreams()
