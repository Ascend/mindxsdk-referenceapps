"""
# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
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
"""

import json
import sys
import os
import argparse
import cv2
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector


parser = argparse.ArgumentParser(description="hello")
parser.add_argument('--url_video', type=str,metavar='PATH',default="rtsp://192.168.88.109:8554/1.264",help='video path.')
parser.add_argument('--label', type=str,default="0",help='ground truth.')
parser.add_argument('--frame_num', type=str,default="40",help='length of video.')
parser.add_argument('--frame_threshold', type=int,default=30,help='threshold of frame num.')
parser.add_argument('--perclos_threshold', type=int,default=0.7,help='threshold of perclos.')
parser.add_argument('--mar_threshold', type=int,default=0.14,help='threshold of mar.')

def get_args(sys_args):
    """
    # obtain the parameters
    # input parameter:(1)sys_args:input variables
    # output parameter:(1)global_args: key-value dictionary of variables.
    """
    global_args = parser.parse_args(sys_args)
    return global_args


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
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

    ret = streamManagerApi.CreateMultipleStreams(pipeline)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # The name of the plugin to get the results from  
    streamName = b"detection"

    keyVec = StringVector()
    keyVec.push_back(b"mxpi_tensorinfer1")
    keyVec.push_back(b"mxpi_videodecoder0")
    keyVec.push_back(b"mxpi_distributor0_0")
    keyVec.push_back(b"mxpi_pfldpostprocess0")
    # Init the list and counting variable
    index = 0
    img_yuv_list = []
    heightAligned_list = []
    widthAligned_list = []
    isFatigue = 0
    YUV_BYTES_NU = 3
    YUV_BYTES_DE = 2
    MARS = []
    frame_num = int(args.frame_num) 
    while True:
        if index == frame_num:
            break
        # Obtain the inference result
        infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)
        # Obtain the PFLD inference results
        tensorList = MxpiDataType.MxpiTensorPackageList()
        tensorList.ParseFromString(infer_result[0].messageBuf)
        ids = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        if ids.shape[0] == 0:
            continue
        # Obtain the PFLD post-processing plugin results
        objectList = MxpiDataType.MxpiObjectList()
        objectList.ParseFromString(infer_result[3].messageBuf)

        MAR = objectList.objectVec[0].x0
        # Obtain the the original image
        visionList = MxpiDataType.MxpiVisionList()
        visionList.ParseFromString(infer_result[1].messageBuf)
        visionData = visionList.visionVec[0].visionData.dataStr
        visionInfo = visionList.visionVec[0].visionInfo
        img_yuv = np.frombuffer(visionData, dtype=np.uint8)
        heightAligned = visionInfo.heightAligned
        widthAligned = visionInfo.widthAligned

        # Add the result of the current frame to the list
        MARS.append(MAR)
        img_yuv_list.append(img_yuv)
        heightAligned_list.append(heightAligned)
        widthAligned_list.append(widthAligned)
        # number of frame
        if len(MARS) >= args.frame_threshold:
            cut_list_num = -1 * args.frame_threshold
            aim_MARS = MARS[cut_list_num:]
            max_index = 0
            max_mar = aim_MARS[0]
            num = 0
            for index_mar, mar in enumerate(aim_MARS):
                # Judge the threshold
                if mar >= args.mar_threshold:
                    num += 1
                if mar > max_mar:
                    max_mar = mar
                    max_index = index_mar
            perclos = num / args.frame_threshold
            # Conform to the fatigue driving conditions
            if perclos >= args.perclos_threshold:
                isFatigue = 1
                print('Fatigue!!!')
            heightAligned_list.pop(0)
            widthAligned_list.pop(0)
            img_yuv_list.pop(0)
        index = index + 1
    # write th result file
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

