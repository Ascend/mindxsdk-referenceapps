# coding=utf-8

"""
Copyright 2021 Huawei Technologies Co., Ltd

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

import time
import sys
from StreamManagerApi import StreamManagerApi, StringVector

if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./pipeline/dirver-detection.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Inputs data to a specified stream based on streamName.
    streamName = b'im_resnet50'
    inPluginId = 0
    retStr = ''
    total_frame = 0
    st_frame = 0
    detect_time = sys.argv[1]
    threshold_1 = 0.2
    threshold_2 = 0.8
    time_start = time.time() 
    while True:        
        # Obtain the inference result by specifying streamName and uniqueId.
        inferResult = streamManagerApi.GetResult(streamName, inPluginId, 10000)
        end = time.time()
        
        if inferResult is None:
            break
        if inferResult.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                inferResult.errorCode, inferResult.data.decode()))
            break
        retStr = inferResult.data.decode()
        cls_id = retStr.split(",")[0].split("{[")[-1]
        if ("0" in cls_id):
            # c0为安全驾驶
            st_frame = st_frame + 1
        
        total_frame = total_frame + 1
                
        if (int(end - time_start) == int(detect_time)):
            thr = st_frame / total_frame
            print("frame_tatal:{}, st_frame:{}, thr:{}".format(total_frame, st_frame, thr))
            print("cls_id:", cls_id)
            if thr < threshold_1:
                print("安全驾驶占比小于阈值，严重警告")
            elif thr >= threshold_1 and thr < threshold_2:
                print("安全驾驶占比小于警告值，注意")
            thr = st_frame = total_frame = 0            
            time_start = time.time()
            
            total_frame = 0
            continue


    # destroy streams
    streamManagerApi.DestroyAllStreams()
