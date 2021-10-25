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
import io
from PIL import Image

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

StreamName = b'galleryImageProcess'

InPluginId = 0

YUV_BYTES_NU = 3
YUV_BYTES_DE = 2

MIN_IMAGE_SIZE = 32
MAX_IMAGE_SIZE = 8192
MIN_IMAGE_WIDTH = 6


def initialize_stream():
    """
    Initialize streams.

    :arg:
        None

    :return:
        Stream api
    """
    streamApi = StreamManagerApi()
    streamState = streamApi.InitManager()
    if streamState != 0:
        errorMessage = "Failed to init Stream manager, streamState=%s" % str(streamState)
        print(errorMessage)
        exit()

    # creating stream based on json strings in the pipeline file: 'ReID.pipeline'
    with open("pipeline/ReID.pipeline", 'rb') as f:
        pipeline = f.read().replace(b'\r', b'').replace(b'\n', b'')
    pipelineString = pipeline

    streamState = streamApi.CreateMultipleStreams(pipelineString)
    if streamState != 0:
        errorMessage = "Failed to create Stream, streamState=%s" % str(streamState)
        print(errorMessage)
        exit()

    return streamApi


def crop_process(streamApi, pluginNameVector, file):
    """
    Crop processing

    :arg:
        streamApi: stream api
        pluginNameVector: the vector of  plugin name

    :return:
        None
    """
    # get infer result
    inferResult = streamApi.GetProtobuf(StreamName, InPluginId, pluginNameVector)

    # checking whether the infer results is valid or not
    if inferResult.size() == 0:
        errorMessage = 'Unable to get effective infer results, please check the stream log for details'
        print(errorMessage)
        exit()
    if inferResult[0].errorCode != 0:
        errorMessage = "GetProtobuf error. errorCode=%d, errorMessage=%s" % (inferResult[0].errorCode,
                                                                             inferResult[0].messageName)
        print(errorMessage)
        exit()

    # the output information of "mxpi_objectpostprocessor0" plugin后处理插件的输出数据
    objectList = MxpiDataType.MxpiObjectList()
    objectList.ParseFromString(inferResult[0].messageBuf)
    # get the crop tensor
    tensorList = MxpiDataType.MxpiVisionList()
    tensorList.ParseFromString(inferResult[1].messageBuf)
    filterImageCount = 0

    for detectedItemIndex in range(0, len(objectList.objectVec)):
        item = objectList.objectVec[detectedItemIndex]
        xLength = int(item.x1) - int(item.x0)
        yLength = int(item.y1) - int(item.y0)
        if xLength < MIN_IMAGE_SIZE or yLength < MIN_IMAGE_WIDTH:
            filterImageCount += 1
            continue
        if item.classVec[0].className == "person":
            cropData = tensorList.visionVec[detectedItemIndex - filterImageCount].visionData.dataStr
            cropInformation = tensorList.visionVec[detectedItemIndex - filterImageCount].visionInfo
            img_yuv = np.frombuffer(cropData, np.uint8)
            img_bgr = img_yuv.reshape(cropInformation.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE,
                                      cropInformation.widthAligned)
            img = cv2.cvtColor(img_bgr, getattr(cv2, "COLOR_YUV2BGR_NV12"))
            cv2.imwrite('./data/cropOwnDataset/{}_{}.jpg'.format(str(file[:-4]),
                                                                 str(detectedItemIndex)), img)


def crop_person_from_own_dataset(imagePath, outputPath, streamApi):
    """
    Crop persons from given images for making your own dataset

    :arg:
        imagePath: the directory of input images
        outputPath: the directory of output
        streamApi: stream api

    :return:
        None
    """
    # constructing the results returned by the stream
    pluginNames = [b"mxpi_objectpostprocessor0", b"mxpi_imagecrop0"]
    pluginNameVector = StringVector()
    for key in pluginNames:
        pluginNameVector.push_back(key)

    # check the file paths
    if os.path.exists(imagePath) != 1:
        errorMessage = 'The ownDataset file does not exist.'
        print(errorMessage)
        exit()
    if os.path.exists(outputPath) != 1:
        errorMessage = 'The cropOwnDataset file does not exist.'
        print(errorMessage)
        exit()
    if len(os.listdir(imagePath)) == 0:
        errorMessage = 'The ownDataset file is empty.'
        print(errorMessage)
        exit()

    # extract the features for all images in query file
    for root, dirs, files in os.walk(imagePath):
        for file in files:
            if file.endswith('.jpg'):
                filePath = os.path.join(root, file)

                queryDataInput = MxDataInput()
                try:
                    image = Image.open(filePath)
                    if image.format != 'JPEG':
                        print('Input image only support jpg')
                        exit()
                    elif image.width < MIN_IMAGE_SIZE or image.width > MAX_IMAGE_SIZE:
                        print('Input image width must in range [32, 8192], curr is {}'.format(image.width))
                        exit()
                    elif image.height < MIN_IMAGE_SIZE or image.height > MAX_IMAGE_SIZE:
                        print('Input image height must in range [32, 8192], curr is {}'.format(image.height))
                        exit()
                    else:
                        # read input image bytes
                        image_bytes = io.BytesIO()
                        image.save(image_bytes, format='JPEG')
                        queryDataInput.data = image_bytes.getvalue()
                except IOError:
                    print('An IOError occurred while opening {}, maybe your input is not a picture'.format(filePath))
                    exit()

                # send the prepared data to the stream
                uniqueId = streamApi.SendData(StreamName, InPluginId, queryDataInput)
                if uniqueId < 0:
                    errorMessage = 'Failed to send data to queryImageProcess stream.'
                    print(errorMessage)
                    exit()

                crop_process(streamApi, pluginNameVector, file)
            else:
                print('Input image only support jpg')
                exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageFilePath', type=str, default='data/ownDataset', help="Input File Path")
    parser.add_argument('--outputFilePath', type=str, default='data/cropOwnDataset', help="Output File Path")
    opt = parser.parse_args()
    streamManagerApi = initialize_stream()
    crop_person_from_own_dataset(opt.imageFilePath, opt.outputFilePath, streamManagerApi)
