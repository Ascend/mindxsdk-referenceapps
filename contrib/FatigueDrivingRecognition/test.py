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
import cv2
import numpy as np
import os
import time
import threading
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
import argparse

parser = argparse.ArgumentParser(description="hello")
parser.add_argument('--url_video', type=str,metavar='PATH',default="rtsp://192.168.88.109:8554/1.264",help='video path.')
parser.add_argument('--label', type=str,default="0",help='ground truth.')
parser.add_argument('--frame_num', type=str,default="40",help='length of video.')

def get_args(sys_args):
  global_args = parser.parse_args(sys_args)
  return global_args

def test_acc(args):
    streamManagerApi = StreamManagerApi()
    # init stream manager
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    pipeline_path = b"pipeline/test.pipeline"
    with open(pipeline_path, 'rb') as f:
        pipelineStr = f.read().decode()
    pipeline = pipelineStr.split("rtsp_url")[0] + args.url_video + pipelineStr.split("rtsp_url")[1]
    pipeline = pipeline.encode()
    ret = streamManagerApi.CreateMultipleStreamsFromFile(pipeline_path)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    time_start=time.time()
    streamName = b"detection"
    
    keyVec = StringVector()
    keyVec.push_back(b"mxpi_tensorinfer1")
    keyVec.push_back(b"mxpi_videodecoder0")
    keyVec.push_back(b"mxpi_distributor0_0")
    keyVec.push_back(b"mxpi_pfldpostprocess0")
    
    img_yuv_list = []
    heightAligned_list = []
    widthAligned_list = []
    isFatigue = 0
    YUV_BYTES_NU = 3
    YUV_BYTES_DE = 2
    MARS = []
    index = 0
    while True:
        if index == 2741:
            break
        infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)

        objectList = MxpiDataType.MxpiObjectList()
        objectList.ParseFromString(infer_result[3].messageBuf)
        MAR = objectList.objectVec[0].x0

        visionList = MxpiDataType.MxpiVisionList()
        visionList.ParseFromString(infer_result[1].messageBuf)
        visionData = visionList.visionVec[0].visionData.dataStr
        visionInfo = visionList.visionVec[0].visionInfo

        img_yuv = np.frombuffer(visionData, dtype=np.uint8)
        heightAligned = visionInfo.heightAligned
        widthAligned = visionInfo.widthAligned
        time_start_calcu = time.time()
        MARS.append(MAR)
        img_yuv_list.append(img_yuv)
        heightAligned_list.append(heightAligned)
        widthAligned_list.append(widthAligned)
        # number of frame
        if len(MARS) >= 30:
            aim_MARS = MARS[-30:]
            max_index = 0
            max_mar = aim_MARS[0]
            num = 0
            for index_mar, mar in enumerate(aim_MARS):
                if mar >= 0.14:
                    num += 1
                if mar > max_mar:
                    max_mar = mar
                    max_index = index_mar
                
            perclos = num / 30
            # threshold
            if perclos >= 0.7:
                isFatigue = 1
                print('Fatigue!!!')

                
        index = index + 1
    f = open("result.txt", "a+")
    if isFatigue == 0:
        print('Normal')
        f.write(args.url_video.split('/')[-1] + ' ' + args.label + ' 0\n')
    else:
        print('Fatigue!!!')
        f.write(args.url_video.split('/')[-1] + ' ' + args.label + ' 1\n')
    f.close()
    # destroy streams
    streamManagerApi.DestroyAllStreams()
    
if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    test_acc(args)