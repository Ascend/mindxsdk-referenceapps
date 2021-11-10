#!/usr/bin/env python
# coding=utf-8

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

import os
import cv2
import numpy as np
import sys
sys.path.append("../proto")
import mxpiRotateobjProto_pb2 as mxpiRotateObjProto
import MxpiDataType_pb2 as MxpiDataType
import argparse

from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

if __name__ == '__main__':
    streamManagerApi = StreamManagerApi()
    # 新建一个流管理StreamManager对象并初始化
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # 构建pipeline
    with open("./pipeline/RotateObjectDetection.pipeline", "rb") as f:
        pipeline_str = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='../image', help='input image path')
    parser.add_argument('--output-path', type=str, default='../detection', help='detection output path')
    parser.add_argument('--labels', action='store_true', default=False, help='whether to print labels')
    
    opt = parser.parse_args()
    print(opt)

    input_path = opt.input_path
    out_path = opt.output_path
    labels = opt.labels
    cont = 0

    if len(os.listdir(input_path)) == 0:
        print("The image folder is empty.") 

    for item in os.listdir(input_path):
        img_path = os.path.join(input_path, item)
        print("Detect the order of the image: {}".format(cont))
        print("detect_file_path:", img_path)
    
        # 构建流的输入对象--检测目标
        dataInput = MxDataInput()
        
        if os.path.exists(input_path) != 1:
            print("The test image does not exist.")

        with open(img_path, 'rb') as f:
            dataInput.data = f.read()

        
        streamName = b'detection'
        inPluginId = 0
        
        # 根据流名将检测目标传入流中
        uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)
        if uniqueId < 0:
            print("Failed to send data to stream.")
            exit()

        keyVec = StringVector()
        keys = (b"mxpi_imagedecoder0", b"mxpi_rotateobjproto")
        for key in keys:
            keyVec.push_back(key)
        
        # get inference result
        infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)
        
        if infer_result.size() == 0:
            print("infer_result is null")
            exit()
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (
                infer_result[0].errorCode))
            exit()

        # mxpi_imagedecoder0 图像解码插件输出信息
        YUV_BYTES_NU = 3
        YUV_BYTES_DE = 2
        visionList = MxpiDataType.MxpiVisionList()
        visionList.ParseFromString(infer_result[0].messageBuf)
        vision_data = visionList.visionVec[0].visionData.dataStr
        visionInfo = visionList.visionVec[0].visionInfo

        # 用输出原件信息初始化OpenCV图像信息矩阵
        img_yuv = np.frombuffer(vision_data, np.uint8)
        img_bgr = img_yuv.reshape(visionInfo.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, 
                                    visionInfo.widthAligned)
        img = cv2.cvtColor(img_bgr, getattr(cv2, "COLOR_YUV2BGR_NV12"))
        
        result_protolist = mxpiRotateObjProto.MxpiRotateobjProtoList()
        result_protolist.ParseFromString(infer_result[1].messageBuf)

        result_protovec = result_protolist.rotateobjProtoVec
        classNum = 16
        rboxes = []
        classID_list = []
        keys = ["x_c", "y_c", "width", "height", "angle", "confidence", "classID", "className"]
        values = [0, 0, 0, 0, 0, 0, 0, 0]
        classNames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
                        'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                        'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 
                        'harbor', 'swimming-pool', 'helicopter', 'container-crane']
        s = ' '
        tl = 1
        tf = tl - 1 if (tl - 1 > 1) else 1
        
        for i in range(len(result_protovec)):
            rbox = dict(zip(keys, values))
            rbox["x_c"] = result_protovec[i].x_c
            rbox["y_c"] = result_protovec[i].y_c
            rbox["width"] = result_protovec[i].width
            rbox["height"] = result_protovec[i].height
            rbox["angle"] = result_protovec[i].angle
            rbox["confidence"] = result_protovec[i].confidence
            rbox["classID"] = result_protovec[i].classID
            rbox["className"] = result_protovec[i].className
            # print(rbox)
            rboxes.append(rbox)
            classID_list.append(result_protovec[i].classID)
        
        for c in np.unique(np.array(classID_list)):
            n = (np.array(classID_list) == c).sum()
            s += '%d %s, ' % (n, classNames[int(c)])
        s = s[:-2] if s else "No object is detectde"
        print("detect result: ", s)
        
        text_out_path = out_path + '/' + 'result_txt/result_before_merge/'
        oriname = item.split('__')[0]
        for rbox in rboxes:
            rect = ((rbox["x_c"], rbox["y_c"]), (rbox["width"], rbox["height"]), rbox["angle"])
            poly = np.float32(cv2.boxPoints(rect))
            poly = np.int0(poly).reshape(8)
            lines = item.split('.')[0] + ' ' + str(round(rbox["confidence"], 3)) + ' ' \
                    + ' '.join(list(map(str, poly))) + ' ' + rbox["className"]
            if not os.path.exists(text_out_path):
                os.makedirs(text_out_path)
            with open(str(text_out_path + oriname) + '.txt', 'a') as f:
                f.writelines(lines + '\n')


        np.random.seed(666)
        colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(classNum)]

        for i in range(len(rboxes)):
            rect = [(rboxes[i]["x_c"], rboxes[i]["y_c"]), (rboxes[i]["width"], 
                        rboxes[i]["height"]), rboxes[i]["angle"]]
            poly = cv2.boxPoints(rect)
            poly = np.int0(poly)
            cv2.drawContours(image=img, contours=[poly], contourIdx=-1, 
                            color=colors[int(rboxes[i]["classID"])], thickness=2 * tl)
            if labels:
                label = '%s %.2f' % (rboxes[i]["className"], rboxes[i]["confidence"])
            else:
                label = '%s' % rboxes[i]["classID"]
            c1 = (int(rboxes[i]["x_c"]), int(rboxes[i]["y_c"]))
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, colors[int(rboxes[i]["classID"])], -1, cv2.LINE_AA)  
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 4, [225, 255, 255], 
                        thickness=tf, lineType=cv2.LINE_AA)
            
        save_path = out_path + '/' + item
        cv2.imwrite(save_path, img)
    
    # destroy streams
    streamManagerApi.DestroyAllStreams()
