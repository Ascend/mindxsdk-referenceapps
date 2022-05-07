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

import os
import sys
import copy
import math
import json
import cv2
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

def get_fileNames(rootdir):
    fs = []
    for root, dirs, files in os.walk(rootdir,topdown = True):
        for name in files: 
            _, ending = os.path.splitext(name)
            if ending == ".jpg":
                fs.append(os.path.join(name))   
    return fs


if __name__ == '__main__':

    testfiles = get_fileNames('./input/')

    if len(testfiles) == 0:
        print("The input directory is EMPTY!")
        print("Please place the picture in './input/' !")
        exit()

    # Create and initialize a new StreamManager object
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # Read and format pipeline file
    with open("./pipeline/identification.pipeline", 'rb') as f:
        pipeline = f.read().replace(b'\r', b'').replace(b'\n', b'')
    pipelineStr = pipeline

    # Create Stream
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    if os.path.exists('result') != True:
        os.mkdir('result')

    for testfile in testfiles:
        testfile = "input/" + testfile

        if os.path.getsize(testfile) == 0:
            print("Error!The test image is empty.")
            continue

        # Create Input Object
        dataInput = MxDataInput()

        with open(testfile, 'rb') as f:
            dataInput.data = f.read()

        # Stream Info
        STREAM_NAME = b'identification'
        INPLUGIN_ID = 0
        # Send Input Data to Stream
        uniqueId = streamManagerApi.SendData(STREAM_NAME, INPLUGIN_ID, dataInput)
        
        print(testfile[6:] + " -------------START-------------")
        print()

        # Get the result returned by the plugins
        keys = [b"mxpi_imagedecoder0", b"mxpi_distributor0_0", b"mxpi_classpostprocessor0"]
        keyVec = StringVector()
        for key in keys:
            keyVec.push_back(key)

        OUTPLUGIN_ID = 0
        infer_result = streamManagerApi.GetProtobuf(STREAM_NAME, OUTPLUGIN_ID, keyVec)
        
        IMGDECODER_INDEX = 0
        YOLO_INDEX = 1
        VEHICLE_INDEX = 2

        # Can not decode the image
        if infer_result.size() == 0:
            print("Error!Please check the input image!")
            continue

        # If only the output of imgdecoder
        if infer_result.size() == 1:
            print("infer_result is null")
            image = cv2.imread(testfile)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_res = copy.deepcopy(image)
            image_res = cv2.cvtColor(image_res, cv2.COLOR_RGB2BGR)
            SRC_PATH = os.path.realpath(__file__).rsplit("/", 1)[0]
            resultname = "./result/" + testfile[6:-4] + "_result.jpg"
            Output_PATH = os.path.join(SRC_PATH, resultname)
            cv2.imwrite(Output_PATH, image_res)
            print()
            print(testfile[6:] + " --------------END--------------")
            print()
            continue

        if infer_result[YOLO_INDEX].errorCode != 0:
            print("GetProtobuf error. errorCode=%d, errorPlugin=%s" % (
                infer_result[YOLO_INDEX].errorCode, infer_result[YOLO_INDEX].messageName))
            continue

        objectList = MxpiDataType.MxpiObjectList()
        objectList.ParseFromString(infer_result[YOLO_INDEX].messageBuf)
        yolo_results = objectList.objectVec

        if infer_result[VEHICLE_INDEX].errorCode != 0:
            print("GetProtobuf error. errorCode=%d, errorPlugin=%s" % (
                infer_result[VEHICLE_INDEX].errorCode, infer_result[VEHICLE_INDEX].messageName))
            continue

        # Get vehicleIdentification result
        classList = MxpiDataType.MxpiClassList()
        classList.ParseFromString(infer_result[VEHICLE_INDEX].messageBuf)
        vehicle_results = classList.classVec
        print(classList)
        
        YUV_BYTES_NU = 3
        YUV_BYTES_DE = 2
        
        # mxpi_imagedecoder0 image decoding output information
        visionList = MxpiDataType.MxpiVisionList()
        visionList.ParseFromString(infer_result[IMGDECODER_INDEX].messageBuf)
        
        vision_data = visionList.visionVec[0].visionData.dataStr
        visionInfo = visionList.visionVec[0].visionInfo

        # Initialize the opencv image information matrix with the output original information
        img_yuv = np.frombuffer(vision_data, np.uint8)

        img_bgr = img_yuv.reshape(visionInfo.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, visionInfo.widthAligned)
        img = cv2.cvtColor(img_bgr, getattr(cv2, "COLOR_YUV2BGR_NV12"))
        
        # Draw image according to vehicle information
        bboxes = []

        # Drawing parameters
        X_OFFSET_PIXEL = 10
        Y_TYPE_OFFSET_PIXEL = 30
        Y_PROB_OFFSET_PIXEL = 60
        TYPE_FONT_SIZE = 0.9
        PROB_FONT_SIZE = 0.8
        FONT_THICKNESS = 2
        RECTANGLE_THICKNESS = 3
        FONT_COLOR = (0, 255, 0)
        RECTANGLE_COLOR = (255, 0, 0)

        # When the confidence of the recognition result is greater than the threshold, it will be marked on the picture
        THRESHOLD = 0.4

        for i, value in enumerate(classList.classVec):
            bboxes = {'x0': int(yolo_results[i].x0),
                    'x1': int(yolo_results[i].x1),
                    'y0': int(yolo_results[i].y0),
                    'y1': int(yolo_results[i].y1),
                    'confidence': round(vehicle_results[i].confidence, 4),
                    'text': vehicle_results[i].className}
            try:
                confidence = bboxes['confidence']
            except KeyError:
                print("Error!Confidence is not included in the results!")
                continue

            if confidence > THRESHOLD:
                
                cv2.putText(img, bboxes['text'], (bboxes['x0'] + X_OFFSET_PIXEL, bboxes['y0'] + Y_TYPE_OFFSET_PIXEL),
                            cv2.FONT_HERSHEY_SIMPLEX, TYPE_FONT_SIZE, FONT_COLOR, FONT_THICKNESS)
                cv2.putText(img, 'prob:' + str(bboxes['confidence']),
                            (bboxes['x0'] + X_OFFSET_PIXEL, bboxes['y0'] + Y_PROB_OFFSET_PIXEL),
                            cv2.FONT_HERSHEY_SIMPLEX, PROB_FONT_SIZE, FONT_COLOR, FONT_THICKNESS)
                cv2.rectangle(img, (bboxes['x0'], bboxes['y0']), (bboxes['x1'], bboxes['y1']),
                            RECTANGLE_COLOR, RECTANGLE_THICKNESS)
            else:
                cv2.putText(img, 'null', (bboxes['x0'] + X_OFFSET_PIXEL, bboxes['y0'] + Y_TYPE_OFFSET_PIXEL), 
                            cv2.FONT_HERSHEY_SIMPLEX, PROB_FONT_SIZE, FONT_COLOR, FONT_THICKNESS)
                cv2.rectangle(img, (bboxes['x0'], bboxes['y0']), (bboxes['x1'], bboxes['y1']),
                            RECTANGLE_COLOR, RECTANGLE_THICKNESS)

        resultfile = "./result/" + testfile[6:-4] + "_result.jpg"
        cv2.imwrite(resultfile, img)
        print(testfile[6:] + " --------------END--------------")
        print()
    # destroy streams
    streamManagerApi.DestroyAllStreams()
