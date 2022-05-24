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

from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
import MxpiDataType_pb2 as MxpiDataType
import numpy as np
import cv2
import os
from PIL import Image

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
    min_image_size = 32
    max_image_size = 8192
    dataInput = MxDataInput()
    filepath = "4.jpg"
    if os.path.exists(filepath) != 1:
        print("Failed to get the input picture. Please check it!")
        streamManagerApi.DestroyAllStreams()
        exit()
    else:
        try:
            image = Image.open(filepath)
            if image.format != 'JPEG':
                print('input image only support jpg, curr format is {}.'.format(image.format))
                streamManagerApi.DestroyAllStreams()
                exit()
            elif image.width < min_image_size or image.width > max_image_size:
                print('input image width must in range [{}, {}], curr width is {}.'.format(
                    min_image_size, max_image_size, image.width))
                streamManagerApi.DestroyAllStreams()
                exit()
            elif image.height < min_image_size or image.height > max_image_size:
                print('input image height must in range [{}, {}], curr height is {}.'.format(
                    min_image_size, max_image_size, image.height))
                streamManagerApi.DestroyAllStreams()
                exit()
            else:
                input_valid = True
                # read input image
                with open(filepath, 'rb') as f:
                    dataInput.data = f.read()
        except IOError:
            print('an IOError occurred while opening {}, maybe your input is not a picture.'.format(filepath))
            streamManagerApi.DestroyAllStreams()
            exit()

    # Inputs data to a specified stream based on streamName.
    streamName = b'segment'
    inPluginId = 0
    uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    # destroy streams  
    streamManagerApi.DestroyAllStreams()