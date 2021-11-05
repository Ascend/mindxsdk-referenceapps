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


import sys
import os
import io
import cv2
import numpy as np
from PIL import Image
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

STREAM_NAME = b'segmentation'
IN_PLUGIN_ID = 0

COLOR_DEPTH = 255

MODEL_OUTPUT_WIDTH = 224
MODEL_OUTPUT_HEIGHT = 224
MODEL_OUTPUT_DIMENSION = 2

BACKGROUND_IMAGE_PATH = sys.argv[1]
PORTRAIT_IMAGE_PATH = sys.argv[2]

DEFAULT_THRESHOLD = 1
EXPECTED_PARAMETERS = 4

REPEAT_AXIS = 3
REPEAT_TIMES = 2

MIN_IMAGE_SIZE = 32
MAX_IMAGE_SIZE = 8192

if __name__ == '__main__':
    #  check input image
    input_path = [BACKGROUND_IMAGE_PATH, PORTRAIT_IMAGE_PATH]
    input_image_data = []
    for i in input_path:
        # check input image
        input_valid = False
        if os.path.exists(i) != 1:
            error_message = 'The {} does not exist'.format(i)
            print(error_message)
        else:
            try:
                image = Image.open(i)
                if image.format != 'JPEG':
                    print('input image only support jpg, curr format is {}'.format(image.format))
                elif image.width < MIN_IMAGE_SIZE or image.width > MAX_IMAGE_SIZE:
                    print('input image width must in range [32, 8192], curr is {}'.format(image.width))
                elif image.height < MIN_IMAGE_SIZE or image.height > MAX_IMAGE_SIZE:
                    print('input image height must in range [32, 8192], curr is {}'.format(image.height))
                else:
                    input_valid = True
                    # read input image bytes
                    image_bytes = io.BytesIO()
                    image.save(image_bytes, format='JPEG')
                    input_image_data.append(image_bytes.getvalue())
            except IOError:
                print('an IOError occurred while opening {}, maybe your input is not a picture'.format(i))
        if not input_valid:
            print('The input image {} is invalid.'.format(i))
            exit()
            
    # initialize the stream manager
    stream_manager = StreamManagerApi()
    stream_state = stream_manager.InitManager()
    if stream_state != 0:
        print("Failed to init Stream manager, ret=%s" % str(stream_state))
        exit()

    # create streams by the pipeline config
    with open("pipeline/segment.pipeline", 'rb') as f:
        pipeline = f.read().replace(b'\r', b'').replace(b'\n', b'')
    pipeline_string = pipeline

    stream_state = stream_manager.CreateMultipleStreams(pipeline_string)
    if stream_state != 0:
        print("Failed to create Stream, ret=%s" % str(stream_state))
        exit()

    # prepare the input of the stream #begin

    # check the background img
    data_input = MxDataInput()
    data_input.data = input_image_data[1]
    # prepare the input of the stream #end

    # send the prepared data to the stream
    unique_id = stream_manager.SendData(STREAM_NAME, IN_PLUGIN_ID, data_input)

    if unique_id < 0:
        print("Failed to send data to stream.")
        exit()

    # construct the resulted streamStateurned by the stream
    plugin_names = [b"mxpi_tensorinfer0"]
    name_vector = StringVector()
    for name in plugin_names:
        name_vector.push_back(name)
    # get inference result
    infer_result = stream_manager.GetProtobuf(STREAM_NAME, 0, name_vector)

    # check whether the inferred results is valid or not
    if len(infer_result) == 0:
        error_message = 'unable to get effective infer results, please check the stream log for details'
        print(error_message)
        exit()
    if infer_result[0].errorCode != 0:
        error_message = "GetProtobuf error. errorCode=%d, errorMessage=%s" % (
        infer_result[0].errorCode, infer_result[0].messageName)
        print(error_message)
        exit()

    # change output tensors into numpy array based on the model's output shape.
    tensor_package = MxpiDataType.MxpiTensorPackageList()
    tensor_package.ParseFromString(infer_result[0].messageBuf)

    # converting the byte data into little-endian 32 bit float array ('<f4')
    output_tensor = np.frombuffer(tensor_package.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
    mask = np.clip((output_tensor * COLOR_DEPTH), 0, COLOR_DEPTH)
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
    mask_resize = cv2.resize(mask, (width, height))
    mask_expansion = np.repeat(mask_resize[..., np.newaxis], REPEAT_AXIS, REPEAT_TIMES)
    result = np.uint8(background * mask_expansion + portrait * (1 - mask_expansion))
    # save the new image in current catalog
    cv2.imwrite('result/result1.jpg', result)
