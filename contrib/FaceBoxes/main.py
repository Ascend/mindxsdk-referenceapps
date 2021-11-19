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

from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
import cv2
import os
import argparse


parser = argparse.ArgumentParser(description='FaceBoxes')
parser.add_argument('--save_folder', default='./data/FDDB_Evaluation/', type=str, help='Dir to save results')
parser.add_argument('--img_info', default='./data/FDDB/img_list.txt')
parser.add_argument('--image_folder', default = './data/FDDB/images/')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
args = parser.parse_args()

if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./pipeline/Faceboxes.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    dataInput = MxDataInput()
    img_addresses = []
    img_names = []
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    fw = open(os.path.join(args.save_folder, 'FDDB_dets.txt'), 'w')

    with open(args.img_info, 'r') as fr:
        for img_address in fr:
            #img_address  e.g. 2002/08/11/big/img_591
            img_addresses.append(os.path.join(args.image_folder, img_address + '.jpg'))

    for i, name_img in enumerate(img_addresses):
        with open(name_img, 'rb')as f:
            dataInput.data = f.read()              
        # Inputs data to a specified stream based on streamName.
        streamName = b'Faceboxes'
        inPluginId = 0
        uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)
        if uniqueId < 0:
            print("Failed to send data to stream.")
            exit()
    
        # Obtain the inference result by specifying streamName and uniqueId.
        keyVec = StringVector()
        keyVec.push_back(b"mxpi_objectpostprocessor0")
        inferResult = streamManagerApi.GetProtobuf(streamName, 0, keyVec)
        fw.write('{:s}\n'.format(img_names[i]))

        if inferResult.size() == 0:
            print("infer_result is null")
            img = cv2.imread(img_addresses[i])
            cv2.imwrite('./data/results/test_{}.jpg'.format(img_names[i]), img)
            fw.write('{:.1f}\n'.format(0))
            continue
    
        # print the infer result
        
        # Obtain mxpi_objectpostprocessor output
        tensorList = MxpiDataType.MxpiObjectList()
        tensorList.ParseFromString(inferResult[0].messageBuf)
        
        fw.write('{:.1f}\n'.format(len(tensorList.objectVec)))
        img = cv2.imread(img_addresses[i])
        for j in range(len(tensorList.objectVec)):
            x0 = tensorList.objectVec[j].x0
            y0 = tensorList.objectVec[j].y0
            x1 = tensorList.objectVec[j].x1
            y1 = tensorList.objectVec[j].y1
            conf = tensorList.objectVec[j].classVec[0].confidence
            # Visualization of results
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
            cv2.putText(img, str(conf), (int(x0), int(y0)+10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            w = x1 - x0 + 1.0
            h = y1 - y0 + 1.0
            fw.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(x0, y0, w, h, conf))
        cv2.imwrite('./data/results/{}.jpg'.format(img_names[i]), img)
    fw.close() 
    # destroy streams
    streamManagerApi.DestroyAllStreams()



