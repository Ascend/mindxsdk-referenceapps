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

import os
import cv2
import sys
import numpy as np
from PIL import Image
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
import MxpiDataType_pb2 as MxpiDataType

if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./pipeline/road.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream & check the input image
    MIN_IMAGE_SIZE = 32
    MAX_IMAGE_SIZE = 8192
    dataInput = MxDataInput()
    if len(sys.argv) != 2:
        print('please input image path')
    else:
        if sys.argv[1] == '':
            print('input image path is not valid, use default config.')
        else:
            FILE_PATH = sys.argv[1]
    if os.path.exists(FILE_PATH) != 1:
        print("Failed to get the input picture. Please check it!")
        streamManagerApi.DestroyAllStreams()
        exit()
    else:
        try:
            image = Image.open(FILE_PATH)
            if image.format != 'JPEG':
                print('input image only support jpg, curr format is {}.'.format(image.format))
                streamManagerApi.DestroyAllStreams()
                exit()
            elif image.width < MIN_IMAGE_SIZE or image.width > MAX_IMAGE_SIZE:
                print('input image width must in range [{}, {}], curr width is {}.'.format(
                    MIN_IMAGE_SIZE, MAX_IMAGE_SIZE, image.width))
                streamManagerApi.DestroyAllStreams()
                exit()
            elif image.height < MIN_IMAGE_SIZE or image.height > MAX_IMAGE_SIZE:
                print('input image height must in range [{}, {}], curr height is {}.'.format(
                    MIN_IMAGE_SIZE, MAX_IMAGE_SIZE, image.height))
                streamManagerApi.DestroyAllStreams()
                exit()
            else:
                INPUT_VALID = True
                # read input image
                with open(FILE_PATH, 'rb') as f:
                    dataInput.data = f.read()
        except IOError:
            print('an IOError occurred while opening {}, maybe your input is not a picture.'.format(FILE_PATH))
            streamManagerApi.DestroyAllStreams()
            exit()

    # Inputs data to a specified stream based on STREAM_NAME.
    STREAM_NAME = b'segment'
    INPLUGIN_ID = 0
    uniqueId = streamManagerApi.SendData(STREAM_NAME, INPLUGIN_ID, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    # destroy streams  
    streamManagerApi.DestroyAllStreams()