#!/usr/bin/env python
# coding=utf-8

# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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

import signal
import os
import sys
import cv2
import numpy as np
import webcolors

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

STANDARD_COLORS = [
    'LawnGreen', 'Chartreuse', 'Aqua', 'Beige', 'Azure', 'BlanchedAlmond', 'Bisque',
    'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'AliceBlue', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def from_colorname_to_bgr(color):
    """
    convert color name to bgr value

    Args:
        color: color name

    Returns: bgr value

    """
    rgb_color = webcolors.name_to_rgb(color)
    result = (rgb_color.blue, rgb_color.green, rgb_color.red)
    return result

def standard_to_bgr(list_color_name):
    """
    generate bgr list from color name list

    Args:
        list_color_name: color name list

    Returns: bgr list

    """
    standard = []
    for i in range(len(list_color_name) - 36):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard
    
def plot_one_box(origin_img, box,color=None, line_thickness=None):
    """
    plot one bounding box on image

    Args:
        origin_img: pending image
        box: infomation of the bounding box
        color: bgr color used to draw bounding box
        line_thickness: line thickness value when drawing the bounding box

    Returns: None

    """
    tl = line_thickness or int(round(0.001 * max(origin_img.shape[0:2])))  # line thickness
    if tl < 1:
        tl = 1
    c1, c2 = (int(box['x0']), int(box['y0'])), (int(box['x1']), int(box['y1']))
    cv2.rectangle(origin_img, c1, c2, color=color, thickness=tl)
    if box['text']:
        tf = max(tl - 2, 1)  # font thickness
        s_size = cv2.getTextSize(str('{:.0%}'.format(box['confidence'])), 0, fontScale=float(tl) / 3, thickness=tf)[0]
        t_size = cv2.getTextSize(box['text'], 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
        cv2.rectangle(origin_img, c1, c2, color, -1)  # filled
        cv2.putText(origin_img, '{}: {:.0%}'.format(box['text'], box['confidence']), (c1[0], c1[1] - 2), 0,
                    float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)

    
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Invaild parameters')
        exit()
    else:
        imagesPath = sys.argv[1]
        if not os.path.exists(imagesPath):
            print("input folder is not exist")
            exit()
        resultPath = sys.argv[2]
        if not os.path.exists(resultPath):
            os.mkdir(resultPath)
    
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
        
    pipeline_path = "picodet.pipeline"
    if os.path.exists(pipeline_path) != 1:
        print("Pipeline does not exist !")
        exit()

    with open(pipeline_path, 'rb') as f:
        pipelineStr = f.read()
        ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
        if ret != 0:
            print("Failed to create Stream, ret=%s" % str(ret))
            exit()  
            
    for root, dirs, images in os.walk(imagesPath, topdown=True):
        if len(images) == 0:
            print("folder ",root, " is empty")
            continue
        for image in images:
            imagePath = os.path.join(root, image)
            print(imagePath)
            dataInput = MxDataInput()
            with open(imagePath, 'rb') as f:
                dataInput.data = f.read()

            streamName = b'detection'
            inPluginId = 0
            uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)
            if uniqueId < 0:
                print("Failed to send data to stream.")
                exit()

            keys = [b"mxpi_objectpostprocessor0",b"mxpi_imagedecoder0"]
            keyVec = StringVector()
            for key in keys:
                keyVec.push_back(key)
            infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)
            if infer_result.size() == 0:
                print("infer_result is null")
                continue

            if infer_result[0].errorCode != 0:
                print("GetProtobuf error. errorCode=%d" % (
                      infer_result[0].errorCode))
                exit()

            YUV_BYTES_NU = 3
            YUV_BYTES_DE = 2
            objectList = MxpiDataType.MxpiObjectList()
            objectList.ParseFromString(infer_result[0].messageBuf)
    
            visionList = MxpiDataType.MxpiVisionList()
            visionList.ParseFromString(infer_result[1].messageBuf)
            vision_data = visionList.visionVec[0].visionData.dataStr
            visionInfo = visionList.visionVec[0].visionInfo
    
            img_yuv = np.frombuffer(vision_data, np.uint8)
            img_bgr = img_yuv.reshape(visionInfo.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, visionInfo.widthAligned)
            img = cv2.cvtColor(img_bgr, getattr(cv2, "COLOR_YUV2BGR_NV12"))
       
            color_list = standard_to_bgr(STANDARD_COLORS)

            for obj in objectList.objectVec:
                box = {'x0': obj.x0,
                       'x1': obj.x1,
                       'y0': obj.y0,
                       'y1': obj.y1,
                       'text': obj.classVec[0].className,
                       'classId': int(obj.classVec[0].classId),
                       'confidence': round(obj.classVec[0].confidence, 4)}
                plot_one_box(img, box, color=color_list[box['classId']])
            cv2.imwrite(os.path.join(resultPath, image), img)
                
    streamManagerApi.DestroyAllStreams()
      
    
    
    
