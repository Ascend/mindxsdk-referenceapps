#!/usr/bin/env python
# -*- encoding: utf-8 -*-
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

import json
import os
import sys
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
import MxpiDataType_pb2 as MxpiDataType
from utils import colorize, calc_psnr
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
DEFAULT_IMAGE_WIDTH = 768
DEFAULT_IMAGE_HEIGHT = 768
SCALE = 3
FONT_SIZE = 16
OFFSET = 5

if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    # create streams by pipeline config file
    pipeline = {
        "superResolution": {
            "stream_config": {
                "deviceId": "0"
            },
            "appsrc0": {
                "props": {
                    "blocksize": "409600"
                },
                "factory": "appsrc",
                "next": "mxpi_imagedecoder0"
            },
            "mxpi_imagedecoder0": {
                "factory": "mxpi_imagedecoder",
                "next": "mxpi_imageresize0"
            },
            "mxpi_imageresize0": {
                "props": {
                    "dataSource": "mxpi_imagedecoder0",
                    "resizeHeight": "256",
                    "resizeWidth": "256"
                },
                "factory": "mxpi_imageresize",
                "next": "mxpi_tensorinfer0"
            },
            "mxpi_tensorinfer0": {
                "props": {
                    "dataSource": "mxpi_imagedecoder0",
                    "modelPath": "model/FSRCNN_256_256.om"
                },
                "factory": "mxpi_tensorinfer",
                "next": "appsink0"
            },
            "appsink0": {
                "props": {
                    "blocksize": "409600"
                },
                "factory": "appsink"
            }
        }
    }
    pipelineStr = json.dumps(pipeline).encode()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    input_image_path = "./image/head.jpg"
    # parse command arguments
    if len(sys.argv) == 2:
        if sys.argv[1] == '':
            print('input image path is valid, use default config.')
        else:
            input_image_path = sys.argv[1]
    # check input image
    if os.path.exists(input_image_path) != 1:
        print("The image image does not exist.")

    image = Image.open(input_image_path).convert('RGB')
    # 768 x 768 high resolution image and 3x reduced image
    hr = image.resize((DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT), resample=Image.BICUBIC)
    lr = hr.resize((hr.width // SCALE, hr.height // SCALE), resample=Image.BILINEAR)
    lr.save("./result/lr.jpg")

    # construct the input of the stream
    dataInput = MxDataInput()
    with open("./result/lr.jpg", 'rb') as f:
        dataInput.data = f.read()
        os.remove("./result/lr.jpg")
    streamName = b'superResolution'
    inPluginId = 0
    key = b"mxpi_tensorinfer0"
    uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()
    keys = [b"mxpi_imagedecoder0", b"mxpi_tensorinfer0"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)

    inferResult = streamManagerApi.GetProtobuf(streamName, 0, keyVec)
    if inferResult.size() == 0:
        print("inferResult is null")
        exit()
    if inferResult[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
            inferResult[0].errorCode, inferResult[0].messageName.decode()))
        exit()
    # get the infer result
    inferList0 = MxpiDataType.MxpiTensorPackageList()
    inferList0.ParseFromString(inferResult[1].messageBuf)
    inferVisionData = inferList0.tensorPackageVec[0].tensorVec[0].dataStr

    inferTensorShape = inferList0.tensorPackageVec[0].tensorVec[0].tensorShape
    output_pic_data = np.frombuffer(inferVisionData, dtype="<f4")
    output_y = colorize(output_pic_data, value_min=None, value_max=None)
    output_y = output_y.reshape(inferTensorShape[2], inferTensorShape[3])
    out_img_y = Image.fromarray(np.uint8(output_y), mode="L")

    hr_YCbCr = hr.convert("YCbCr")
    old_y, cb, cr = hr_YCbCr.split()
    out_img = Image.merge("YCbCr", [out_img_y, cb, cr]).convert("RGB")

    # calculate peak signal-to-noise ratio
    PSNR = calc_psnr(out_img_y, old_y)
    print('PSNR: {:.2f}'.format(PSNR))

    # create canvas for finished drawing
    target = Image.new('RGB', (lr.width + OFFSET + out_img.width, out_img.height + OFFSET * 5), "white")
    # splice the pictures line by line
    target.paste(lr, (0, out_img.height - lr.height))
    target.paste(out_img, (lr.width + OFFSET, 0))
    font_set = {
        "type": "./font/SourceHanSansCN-Normal-2.otf",
        "size": FONT_SIZE,
        "color": (0, 0, 0),
        "psnr_content": 'PSNR: {:.2f}'.format(PSNR),
        "origin_content": '输入图像',
        "infer_content": '输出图像',
        "psnr_location": (0, out_img.height - lr.height - OFFSET * 6),
        "origin_location": (lr.width // 2 - OFFSET * 10, out_img.height + OFFSET * 2),
        "infer_location": (lr.width * 2 + OFFSET * 20, out_img.height + OFFSET * 2),
    }
    # create a brush to write text to the picture
    draw = ImageDraw.Draw(target)
    # set font type and size
    font = ImageFont.truetype(font_set["type"], font_set["size"])
    # draw into the picture according to the position, content, color and font
    draw.text(font_set["psnr_location"], font_set["psnr_content"], font_set["color"], font=font)
    draw.text(font_set["origin_location"], font_set["origin_content"], font_set["color"], font=font)
    draw.text(font_set["infer_location"], font_set["infer_content"], font_set["color"], font=font)
    # save visualization results
    _, fileName = os.path.split(input_image_path)
    out_path = "./result/" + fileName
    target.save(out_path, quality=50)
    # destroy streams
    streamManagerApi.DestroyAllStreams()