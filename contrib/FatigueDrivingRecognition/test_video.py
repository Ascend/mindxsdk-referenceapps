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
parser.add_argument('--frame_num', type=int,default=40,help='length of video.')
parser.add_argument('--frame_threshold', type=int,default=30,help='threshold of frame num.')
parser.add_argument('--perclos_threshold', type=int,default=0.7,help='threshold of perclos.')
parser.add_argument('--mar_threshold', type=int,default=0.14,help='threshold of mar.')
parser.add_argument('--online_flag', type = bool, default = False, help = 'if the video is online.')

def get_args(sys_args):
    """
    # obtain the parameters
    # input parameter:
    #         sys_args:input variables
    # return: 
    #         global_args: key-value dictionary of variables.
    """
    global_args = parser.parse_args(sys_args)
    return global_args



if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    frame_num = args.frame_num
    streamManagerApi = StreamManagerApi()
    # init stream manager
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    pipeline_path = b"pipeline/test_video.pipeline"
    ret = streamManagerApi.CreateMultipleStreamsFromFile(pipeline_path)
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
    img_yuv_list = []
    heightAligned_list = []
    widthAligned_list = []
    isFatigue = 0
    nobody_flag = 0
    nobody_num = 0
    nobody_threshold = 30
    YUV_BYTES_NU = 3
    YUV_BYTES_DE = 2
    MARS = []
    index = 0
    position = (5, 50)
    color = (0, 0, 255)
    font_weight = 2
    font_size = 1.3
    err_code = 2017
    while True:
        if index == frame_num and args.online_flag == False:
            break
        # Obtain the inference result
        infer = streamManagerApi.GetResult(streamName, b'appsink0', keyVec)
        if infer.errorCode == err_code:
            index = index + 1
            nobody_num = nobody_num + 1
            if nobody_num >= nobody_threshold:
                nobody_flag = 1
                print('Nobody is detected.')
            continue
        if infer.errorCode != 0:
            print("GetResult error. errorCode=%d, errorMsg=%s" % (infer.errorCode, infer.errorMsg))
        nobody_num = 0
        # Obtain the PFLD post-processing plugin results
        infer_result_3 = infer.metadataVec[3]
        objectList = MxpiDataType.MxpiObjectList()
        objectList.ParseFromString(infer_result_3.serializedMetadata)
        MAR = objectList.objectVec[0].x0
        # Obtain the the original image
        infer_result_1 = infer.metadataVec[1]
        visionList = MxpiDataType.MxpiVisionList()
        visionList.ParseFromString(infer_result_1.serializedMetadata)
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
                
            # Calculate percentage
            perclos = num / args.frame_threshold
            # Conform to the fatigue driving conditions
            if perclos >= args.perclos_threshold:
                isFatigue = 1
                print('Fatigue!!!')
                # Visual result
                img_yuv_fatigue = img_yuv_list[max_index]
                img_yuv_fatigue = img_yuv_fatigue.reshape(heightAligned_list[max_index] * YUV_BYTES_NU // YUV_BYTES_DE,widthAligned_list[max_index])
                img_fatigue = cv2.cvtColor(img_yuv_fatigue, cv2.COLOR_YUV2BGR_NV12)
                cv2.putText(img_fatigue, "Warning!!! Fatigue!!!", position, cv2.FONT_HERSHEY_SIMPLEX, font_size, color, font_weight)
                index_print = index - args.frame_threshold + max_index
                image_path = "fatigue/"
                if not os.path.exists(image_path):
                    os.mkdir(image_path)
                image_name = image_path + str(index_print) + ".jpg"
                cv2.imwrite(image_name, img_fatigue)
            heightAligned_list.pop(0)
            widthAligned_list.pop(0)
            img_yuv_list.pop(0)
        
        index = index + 1
    # Output result
    if nobody_flag == 1:
        print('No one was detected for some time.')
    if isFatigue == 0:
        print('Normal')
    else:
        print('Fatigue!!!')
    # destroy streams
    streamManagerApi.DestroyAllStreams()