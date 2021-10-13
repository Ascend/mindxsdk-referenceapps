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

import argparse
import os
import cv2
import numpy as np

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

StreamName = b'galleryImageProcess'

InPluginId = 0

YUV_BYTES_NU = 3
YUV_BYTES_DE = 2


# initialize streams:
def initialize_stream():
    streamApi = StreamManagerApi()
    streamState = streamApi.InitManager()
    if streamState != 0:
        errorMessage = "Failed to init Stream manager, streamState=%s" % str(streamState)
        raise AssertionError(errorMessage)

    # creating stream based on json strings in the pipeline file: 'ReID.pipeline'
    with open("pipeline/ReID.pipeline", 'rb') as f:
        pipeline = f.read().replace(b'\r', b'').replace(b'\n', b'')
    pipelineString = pipeline

    streamState = streamApi.CreateMultipleStreams(pipelineString)
    if streamState != 0:
        errorMessage = "Failed to create Stream, streamState=%s" % str(streamState)
        raise AssertionError(errorMessage)

    return streamApi


# Crop persons from given images for making your own dataset
def crop_person_from_own_dataset(imagePath, outputPath, streamApi):

    # constructing the results returned by the stream
    pluginNames = [b"mxpi_objectpostprocessor0", b"mxpi_imagecrop0"]
    pluginNameVector = StringVector()
    for key in pluginNames:
        pluginNameVector.push_back(key)

    # check the file paths
    if os.path.exists(imagePath) != 1:
        errorMessage = 'The query file does not exist.'
        raise AssertionError(errorMessage)
    if os.path.exists(outputPath) != 1:
        errorMessage = 'The query file does not exist.'
        raise AssertionError(errorMessage)
    if len(os.listdir(imagePath)) == 0:
        errorMessage = 'The query file is empty.'
        raise AssertionError(errorMessage)

    # extract the features for all images in query file
    for root, dirs, files in os.walk(imagePath):
        for file in files:
            if file.endswith('.jpg'):

                queryDataInput = MxDataInput()
                filePath = os.path.join(root, file)
                with open(filePath, 'rb') as f:
                    queryDataInput.data = f.read()

                # send the prepared data to the stream
                uniqueId = streamApi.SendData(StreamName, InPluginId, queryDataInput)
                if uniqueId < 0:
                    errorMessage = 'Failed to send data to queryImageProcess stream.'
                    raise AssertionError(errorMessage)

                # get infer result
                inferResult = streamApi.GetProtobuf(StreamName, InPluginId, pluginNameVector)

                # checking whether the infer results is valid or not
                if inferResult.size() == 0:
                    errorMessage = 'unable to get effective infer results, please check the stream log for details'
                    raise IndexError(errorMessage)
                if inferResult[0].errorCode != 0:
                    errorMessage = "GetProtobuf error. errorCode=%d, errorMessage=%s" % (inferResult[0].errorCode,
                                                                                         inferResult[0].messageName)
                    raise AssertionError(errorMessage)

                # the output information of "mxpi_objectpostprocessor0" plugin后处理插件的输出数据
                objectList = MxpiDataType.MxpiObjectList()
                objectList.ParseFromString(inferResult[0].messageBuf)
                # get the crop tensor
                tensorList = MxpiDataType.MxpiVisionList()
                tensorList.ParseFromString(inferResult[1].messageBuf)
                for detectedItemIndex in range(0, len(objectList.objectVec)):
                    item = objectList.objectVec[detectedItemIndex]
                    if item.classVec[0].className == "person":
                        cropData = tensorList.visionVec[detectedItemIndex].visionData.dataStr
                        cropInformation = tensorList.visionVec[detectedItemIndex].visionInfo
                        img_yuv = np.frombuffer(cropData, np.uint8)
                        img_bgr = img_yuv.reshape(cropInformation.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE,
                                                  cropInformation.widthAligned)
                        img = cv2.cvtColor(img_bgr, getattr(cv2, "COLOR_YUV2BGR_NV12"))
                        cv2.imwrite('./data/cropOwnDataset/{}_{}.jpg'.format(str(file[:-4]),
                                                                             str(detectedItemIndex)), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageFilePath', type=str, default='data/ownDataset', help="Query File Path")
    parser.add_argument('--outputFilePath', type=str, default='data/cropOwnDataset', help="Gallery File Path")
    opt = parser.parse_args()
    streamManagerApi = initialize_stream()
    crop_person_from_own_dataset(opt.imageFilePath, opt.outputFilePath, streamManagerApi)

