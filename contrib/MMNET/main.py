
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
    with open("./pipeline/MMNET.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
        
    # Construct the input of the stream & check the input image
    min_image_size = 32
    max_image_size = 8192
    dataInput = MxDataInput()
    filepath = "test.jpg"
    if os.path.exists(filepath) != 1:
        print("Failed to get the input picture. Please check it!")
        exit()
    else:
        try:
            image = Image.open(filepath)
            if image.format != 'JPEG':
                print('input image only support jpg, curr format is {}.'.format(image.format))
                exit()
            elif image.width < min_image_size or image.width > max_image_size:
                print('input image width must in range [{}, {}], curr width is {}.'.format(
                    min_image_size, max_image_size, image.width))
                exit()
            elif image.height < min_image_size or image.height > max_image_size:
                print('input image height must in range [{}, {}], curr height is {}.'.format(
                    min_image_size, max_image_size, image.height))
                exit()
            else:
                input_valid = True
                # read input image
                with open(filepath, 'rb') as f:
                    dataInput.data = f.read()
        except IOError:
            print('an IOError occurred while opening {}, maybe your input is not a picture.'.format(filepath))
            exit()

    # Inputs data to a specified stream based on streamName.
    streamName = b'mmnet'
    inPluginId = 0
    uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    keyVec = StringVector()
    keyVec.push_back(b"mxpi_tensorinfer0")
    infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)
    # print the infer result
    if infer_result.size() == 0:
        print("infer_result is null")
        exit()

    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
            infer_result[0].errorCode, infer_result[0].data.decode()))
        exit()

    tensorList = MxpiDataType.MxpiTensorPackageList()
    tensorList.ParseFromString(infer_result[0].messageBuf)
    prediction = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype = np.float32)
    prediction_shape = tensorList.tensorPackageVec[0].tensorVec[0].tensorShape
    prediction = np.reshape(prediction, prediction_shape)

    out_1 = np.reshape(prediction[:, :, :, 1], (256, 256))
    out_1 = 255 * out_1

    # Resize the model output image (256, 256) to the original size
    img = cv2.imread(filepath)
    size = img.shape
    out_1 = cv2.resize(out_1, (size[1], size[0]))
    
    filepath_out = "test-out.jpg"
    cv2.imwrite(filepath_out, out_1)

    # destroy streams
streamManagerApi.DestroyAllStreams()
