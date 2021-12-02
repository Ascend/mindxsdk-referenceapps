#!/usr/bin/env python
# coding=utf-8

"""
Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.

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
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
import cv2
import os
import argparse
from PIL import Image


parser = argparse.ArgumentParser(description='FaceBoxes')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--image_path', default='./test.jpg')
parser.add_argument('--save_image_path', default='./testresult.jpg')
args = parser.parse_args()

if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file

    with open("./Faceboxes.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    dataInput = MxDataInput()
    # test whether test image satisfies requirements or not
    min_image_size = 32
    max_image_size = 8192
    if os.path.exists(args.image_path) != 1:
        print('The {} does not exist.'.format(args.image_path))
    else:
        try:
            image = Image.open(args.image_path)
            if image.format != 'JPEG':
                print('input image only support jpg, curr format is {}.'.format(image.format))
            elif image.width < min_image_size or image.width > max_image_size:
                print('input image width must in range [{}, {}], curr width is {}.'.format(
                    min_image_size, max_image_size, image.width))
            elif image.height < min_image_size or image.height > max_image_size:
                print('input image height must in range [{}, {}], curr height is {}.'.format(
                    min_image_size, max_image_size, image.height))
        except IOError:
            print('an IOError occurred while opening {}, maybe your input is not a picture.'.format(args.image_path))

    with open(args.image_path, 'rb') as f:
        dataInput.data = f.read()
        # Inputs data to a specified stream based on streamName.
    streamName = b'Faceboxes'
    inPluginId = 0
    uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    # Obtain the inference result by specifying streamName and uniqueId.
    keyVec = StringVector()
    keyVec.push_back(b"mxpi_objectpostprocessor0")
    inferResult = streamManagerApi.GetProtobuf(streamName, 0, keyVec)

    if inferResult.size() == 0:
        print("infer_result is null")
        img = cv2.imread(args.image_path)
        cv2.imwrite(args.save_image_path, img)

    # print the infer result

    # Obtain mxpi_objectpostprocessor output
    tensorList = MxpiDataType.MxpiObjectList()
    tensorList.ParseFromString(inferResult[0].messageBuf)

    img = cv2.imread(args.image_path)
    for j in range(len(tensorList.objectVec)):
        x0 = tensorList.objectVec[j].x0
        y0 = tensorList.objectVec[j].y0
        x1 = tensorList.objectVec[j].x1
        y1 = tensorList.objectVec[j].y1
        conf = tensorList.objectVec[j].classVec[0].confidence
        # Visualization of results
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
        cv2.putText(img, str(conf), (int(x0), int(y0) + 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        w = x1 - x0 + 1.0
        h = y1 - y0 + 1.0

    cv2.imwrite(args.save_image_path, img)

# destroy streams
streamManagerApi.DestroyAllStreams()



