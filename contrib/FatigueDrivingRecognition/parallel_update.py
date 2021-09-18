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
import time
import threading
import cv2
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

index = 0
index_second = 0

parser = argparse.ArgumentParser(description="hello")
parser.add_argument('--limit_of_time', type=int,default=10,help='time of test.')
parser.add_argument('--frame_num_1', type=int,default=40,help='length of video1.')
parser.add_argument('--frame_num_2', type=int,default=40,help='length of video2.')
parser.add_argument('--frame_threshold', type=int,default=30,help='threshold of frame num.')
parser.add_argument('--perclos_threshold', type=int,default=0.7,help='threshold of perclos.')
parser.add_argument('--mar_threshold', type=int,default=0.14,help='threshold of mar.')

def fun_timer(time_limit):
    """
    # set timer
    # input parameter:
    #       time:Length of time to be measured
    # return: 
    #       void
    """
    print("frame_num", index + index_second)
    speed = (index + index_second) / time_limit
    print("speed:", speed)
    f = open("performance.txt", "w")
    str1 = "Time:" + str(time_limit) + "s\n"
    str2 = "Speed:" + str(speed) + "fps\n"
    f.write(str1)
    f.write(str2)
    f.close()

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
    limit_time = args.limit_of_time
    frame_num1 = args.frame_num_1
    frame_num2 = args.frame_num_2
    streamManagerApi = StreamManagerApi()
    # init stream manager
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    pipeline_path = b"pipeline/parallel_pipeline.pipeline"
    ret = streamManagerApi.CreateMultipleStreamsFromFile(pipeline_path)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    # The name of the plugin to get the results from  
    streamName = b"detection"
    
    keyVec = StringVector()
    keyVec.push_back(b"ReservedFrameInfo")
    keyVec.push_back(b"mxpi_tensorinfer1")
    keyVec.push_back(b"mxpi_videodecoder0")
    keyVec.push_back(b"mxpi_distributor0_0")
    keyVec.push_back(b"mxpi_pfldpostprocess0")
    keyVec.push_back(b"mxpi_videodecoder1")

    # Init the list and counting variable
    img_yuv_list = []
    heightAligned_list = []
    widthAligned_list = []
    isFatigue = 0
    img_yuv_list_1 = []
    isFatigue_1 = 0
    YUV_BYTES_NU = 3
    YUV_BYTES_DE = 2
    img_yuv_list_2 = []
    heightAligned_list_2 = []
    widthAligned_list_2 = []
    frame_num_list = [frame_num1, frame_num2]
    MARS_1 = []
    MARS_2 = []
    
    while True:
        if index == frame_num_list[0] and index_second == frame_num_list[1]:
            break
        # Set timer
        if index == 0 and index_second == 0:
            timer = threading.Timer(limit_time, fun_timer, (limit_time, ))
            timer.start()
        # Obtain the inference result
        infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)
        # Obtain the PFLD inference results
        tensorList = MxpiDataType.MxpiTensorPackageList()
        tensorList.ParseFromString(infer_result[1].messageBuf)
        ids = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        if ids.shape[0] == 0:
            continue
        # Obtain the frame information
        FrameList0 = MxpiDataType.MxpiFrameInfo()
        FrameList0.ParseFromString(infer_result[0].messageBuf)
        if index == frame_num_list[0] and FrameList0.channelId == 0:
            continue
        elif index_second == frame_num_list[1] and FrameList0.channelId == 1:
            continue
        # Channel 0
        if FrameList0.channelId == 0:
            # Obtain the PFLD post-processing plugin results
            objectList = MxpiDataType.MxpiObjectList()
            objectList.ParseFromString(infer_result[4].messageBuf)
            MAR = objectList.objectVec[0].x0
            height_left = objectList.objectVec[0].y0
            height_right = objectList.objectVec[0].y1
            # Obtain the the original image
            visionList = MxpiDataType.MxpiVisionList()
            visionList.ParseFromString(infer_result[2].messageBuf)
            visionData = visionList.visionVec[0].visionData.dataStr
            visionInfo = visionList.visionVec[0].visionInfo
            img_yuv = np.frombuffer(visionData, dtype=np.uint8)
            heightAligned = visionInfo.heightAligned
            widthAligned = visionInfo.widthAligned
            # Add the result of the current frame to the list
            MARS_1.append(MAR)
            img_yuv_list.append(img_yuv)
            heightAligned_list.append(heightAligned)
            widthAligned_list.append(widthAligned)
            # number of frame
            if len(MARS_1) >= args.frame_threshold:
                cut_list_num = -1 * args.frame_threshold
                aim_MARS = MARS_1[cut_list_num:]
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
                    # visualization
                    img_yuv_fatigue = img_yuv_list[max_index]
                    h = heightAligned_list[max_index] * YUV_BYTES_NU // YUV_BYTES_DE
                    w = widthAligned_list[max_index]
                    img_yuv_fatigue = img_yuv_fatigue.reshape(h, w)
                    img_fatigue = cv2.cvtColor(img_yuv_fatigue, cv2.COLOR_YUV2BGR_NV12)
                    cv2.putText(img_fatigue, "Warning!!! Fatigue!!!", (5, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 2)
                    index_print = index - 30 + max_index
                    image_path = "fatigue/"
                    if not os.path.exists(image_path):
                        os.mkdir(image_path)
                    image_path = "fatigue/0/"
                    if not os.path.exists(image_path):
                        os.mkdir(image_path)
                    image_name = image_path + str(index_print) + ".jpg"
                    cv2.imwrite(image_name, img_fatigue)
                heightAligned_list.pop(0)
                widthAligned_list.pop(0)
                img_yuv_list.pop(0)


            index = index + 1

        # Channel 0
        elif FrameList0.channelId == 1:
            # Obtain the PFLD post-processing plugin results
            objectList = MxpiDataType.MxpiObjectList()
            objectList.ParseFromString(infer_result[3].messageBuf)
            MAR = objectList.objectVec[0].x0
            height_left = objectList.objectVec[0].y0
            height_right = objectList.objectVec[0].y1
            # Obtain the the original image
            visionList = MxpiDataType.MxpiVisionList()
            visionList.ParseFromString(infer_result[4].messageBuf)
            visionData = visionList.visionVec[0].visionData.dataStr
            visionInfo = visionList.visionVec[0].visionInfo
            

            img_yuv = np.frombuffer(visionData, dtype=np.uint8)
            heightAligned = visionInfo.heightAligned
            widthAligned = visionInfo.widthAligned
            # Add the result of the current frame to the list
            MARS_2.append(MAR)
            img_yuv_list_2.append(img_yuv)
            heightAligned_list_2.append(heightAligned)
            widthAligned_list_2.append(widthAligned)
            # Judge the threshold
            if len(MARS_2) >= args.frame_threshold:
                cut_list_num = -1 * args.frame_threshold
                aim_MARS = MARS_2[cut_list_num:]
                max_index = 0
                max_mar = aim_MARS[0]
                num = 0
                for index_mar, mar in enumerate(aim_MARS):
                    if mar >= args.mar_threshold:
                        num += 1
                    if mar < max_mar:
                        max_mar = mar
                        max_index = index_mar
                
                perclos = num / args.frame_threshold
                
                # Conform to the fatigue driving conditions
                if perclos >= args.perclos_threshold:
                    isFatigue_1 = 1
                    print('Fatigue!!!')
                    # visualization
                    img_yuv_fatigue = img_yuv_list_2[max_index]
                    h = heightAligned_list_2[max_index] * YUV_BYTES_NU // YUV_BYTES_DE
                    w = widthAligned_list_2[max_index]
                    img_yuv_fatigue = img_yuv_fatigue.reshape(h, w)
                    img_fatigue = cv2.cvtColor(img_yuv_fatigue, cv2.COLOR_YUV2BGR_NV12)
                    cv2.putText(img_fatigue, "Warning!!! Fatigue!!!", (5, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 2)
                    index_print = index_second - 30 + max_index
                    image_path = "fatigue/"
                    if not os.path.exists(image_path):
                        os.mkdir(image_path)
                    image_path = "fatigue/1/"
                    if not os.path.exists(image_path):
                        os.mkdir(image_path)
                    image_name = image_path + str(index_print) + ".jpg"
                    cv2.imwrite(image_name, img_fatigue)
                heightAligned_list_2.pop(0)
                widthAligned_list_2.pop(0)
                img_yuv_list_2.pop(0)


            index_second = index_second + 1

    # print the result
    if isFatigue == 0:
        print('Normal')
        
    else:
        print('Fatigue!!!')
    if isFatigue_1 == 0:
        print('1 is Normal')
        
    else:
        print('1 is Fatigue!!!')

    # destroy streams
    streamManagerApi.DestroyAllStreams()
