#!/usr/bin/env python
# coding=utf-8

"""
 Copyright 2020 Huawei Technologies Co., Ltd

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

from StreamManagerApi import *
import json
import cv2
import numpy as np

if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("../pipeline/IDCardRecognition.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    dataInput = MxDataInput()
    
    img_path = "../data/idcard/test1.jpg"
   
    with open(img_path, 'rb') as f:
        dataInput.data = f.read()
    
    # Inputs data to a specified stream based on streamName.
    streamName = b'IDCardRecognition'
    inPluginId = 0
    uniqueId = streamManagerApi.SendDataWithUniqueId(streamName, inPluginId, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    # Obtain the inference result by specifying streamName and uniqueId.
    inferResult = streamManagerApi.GetResultWithUniqueId(streamName, uniqueId, 3000)
    if inferResult.errorCode != 0:
        print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
            inferResult.errorCode, inferResult.data.decode()))
        exit()

    results = json.loads(inferResult.data.decode())
    print(results)
    bboxes = []
    for bbox in results['MxpiTextObject']:
        bboxes.append({'x0': int(bbox['x0']),
                       'x1': int(bbox['x1']),
                       'x2': int(bbox['x2']),
                       'x3': int(bbox['x3']),
                       'y0': int(bbox['y0']),
                       'y1': int(bbox['y1']),
                       'y2': int(bbox['y2']),
                       'y3': int(bbox['y3']),
                       'confidence': round(bbox['confidence'], 4),
                       'text': bbox['MxpiTextsInfo'][0]['text']    
                   })
    
    from PIL import Image, ImageDraw, ImageFont
    img = Image.open(img_path)
    draw = ImageDraw.ImageDraw(img)
    for bbox in bboxes:
        draw.polygon([(bbox['x0'], bbox['y0']), (bbox['x1'], bbox['y1']), 
            (bbox['x2'], bbox['y2']), (bbox['x3'], bbox['y3'])], outline=(255, 0, 0))
        fontStyle = ImageFont.truetype("heiTC-Bold.otf", 13, encoding="utf-8")
        text = ""
        for item in bbox['text']:
            text += item 
        draw.text((bbox['x0'], bbox['y0']-16), text, (255, 0, 0), font=fontStyle)
       
    out_path="../data/out_idcard/" + img_path.split('/')[-1]
    img.save(out_path)
    
    # destroy streams
    streamManagerApi.DestroyAllStreams()
