#!/usr/bin/env python
#-*-coding:utf-8-*-

"""
Portrait Segmentation and Background Replacement
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


import os
import sys
import numpy as np
import cv2
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

STREAM_NAME = b'segmentation'
IN_PLUGIN_ID = 0

COLOR_DEPTH = 255

MODEL_OUTPUT_WIDTH = 224
MODEL_OUTPUT_HEIGHT = 224
MODEL_OUTPUT_DIMENSION = 2

DEFAULT_THRESHOLD = 1
EXPECTED_PARAMETERS = 4

REPEAT_AXIS = 3
REPEAT_TIMES = 2

if __name__ == '__main__':

    # initialize the stream manager
    streamManager = StreamManagerApi()
    streamState = streamManager.InitManager()
    if streamState != 0:
        errrorMessage = "Failed to init Stream manager, streamState=%s" % str(streamState)
        raise AssertionError(errrorMessage)

    # create streams by the pipeline config
    with open("pipeline/segment.pipeline", 'rb') as f:
        pipeline = f.read().replace(b'\r', b'').replace(b'\n', b'')
    pipelineString = pipeline

    streamState = streamManager.CreateMultipleStreams(pipelineStr)
    if streamState != 0:
        errorMessage ="Failed to create Stream, streamState=%s" % str(streamState)
        raise AssertionError(errorMessage)

    # prepare the input of the stream #begin

    # check the background img
    dataInput = MxDataInput()
    if os.path.exists(sys.argv[1]) != 1:
        errorMessage = 'The background image does not exist.'
        raise AssertionError(errorMessage)
    # check the portrait img
    if os.path.exists(sys.argv[2]) != 1:
        errorMessage = 'The portrait image does not exist.'
        raise AssertionError(errorMessage)
    with open(sys.argv[2], 'rb') as f:
        dataInput.data = f.read()
    # prepare the input of the stream #end

    # send the prepared data to the stream
    uniqueId = streamManager.SendData(STREAM_NAME, IN_PLUGIN_ID, dataInput)

    if uniqueId < 0:
        errorMessage = 'Failed to send data to stream.'
        raise AssertionError(errorMessage)

    # construct the resulted streamStateurned by the stream
    keys = [b"mxpi_tensorinfer0"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)
    # get inference result
    inferResult = streamManager.GetProtobuf(STREAM_NAME, 0, keyVec)

    # check whether the inferred results is valid or not
    if len(inferResult) == 0:
        errorMessage = 'unable to get effective infer results, please check the stream log for details'
        raise IndexError(errorMessage)
    if inferResult[0].errorCode != 0:
        errorMessage = "GetProtobuf error. errorCode=%d, errorMessage=%s" % (
        inferResult[0].errorCode, inferResult[0].messageName)
        raise AssertionError(errorMessage)

    # change output tensors into numpy array based on the model's output shape.
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(inferResult[0].messageBuf)

    # converting the byte data into little-endian 32 bit float array ('<f4')
    outputTensor = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
    mask = np.clip((outputTensor * COLOR_DEPTH), 0, COLOR_DEPTH)
    mask = mask.reshape(MODEL_OUTPUT_WIDTH, MODEL_OUTPUT_HEIGHT, MODEL_OUTPUT_DIMENSION)
    mask = mask[:, :, 0]

    # a threshold t to determine how precise the segmentation strategy will be, a lower of value of t will only
    # remains pixels that are more likely to be a portrait. The default setting of t is 1, which remains all the pixels
    # that are classified as the portrait by the PortraitNet model.
    threshold = float(sys.argv[3] if len(sys.argv) == EXPECTED_PARAMETERS else DEFAULT_THRESHOLD)
    mask[mask >= (threshold * COLOR_DEPTH)] = 255

    # read the background and portrait image
    background = cv2.imread(sys.argv[1])
    portrait = cv2.imread(sys.argv[2])
    height, width = portrait.shape[:2]
    # resize the background image based on the size of portrait image
    background = cv2.resize(background, (width, height))
    # replace the background in portrait image with the new background
    mask = mask / COLOR_DEPTH
    maskResize = cv2.resize(mask, (width, height))
    maskExpansion = np.repeat(maskResize[..., np.newaxis], REPEAT_AXIS, REPEAT_TIMES)
    result = np.uint8(background * maskExpansion + portrait * (1 - maskExpansion))
    # save the new image in current catalog
    cv2.imwrite('result/result.jpg', result)
