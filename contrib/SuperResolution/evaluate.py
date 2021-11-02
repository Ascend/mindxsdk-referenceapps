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
# Unless required by applicable law or agreimport json
import json
import os
import io
import sys
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
import MxpiDataType_pb2 as MxpiDataType
from utils import colorize, calc_psnr
import numpy as np
from PIL import Image
DEFAULT_IMAGE_WIDTH = 768
DEFAULT_IMAGE_HEIGHT = 768
SCALE = 3
FONT_SIZE = 16
OFFSET_5 = 5
OFFSET_20 = 20


def infer(input_image_path, streamManagerapi):
	"""
	image super-resolution inference
	:param input_image_path: input image path
	:param streamManagerapi: streamManagerapi
	:return: no return
	"""
    if os.path.exists(input_image_path) != 1:
        print("The input image does not exist.")
        exit()
    image = Image.open(input_image_path).convert('RGB')
    # high resolution image, (that is ground true)
    hr = image.resize((DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT), resample=Image.BICUBIC)
    # low resolution image
    lr = hr.resize((hr.width // SCALE, hr.height // SCALE), resample=Image.BILINEAR)
    # interpolated low-resolution image
    ilr = lr.resize((lr.width * SCALE, lr.height * SCALE), resample=Image.BICUBIC)

    # construct the input of the stream
    ilr_image_bytes = io.BytesIO()
    ilr.save(ilr_image_bytes, format='JPEG')
    input_image_data = ilr_image_bytes.getvalue()
    dataInput = MxDataInput()
    dataInput.data = input_image_data

    # inputs data to a specified stream based on streamName
    streamName = b'superResolution'
    inPluginId = 0
    uniqueId = streamManagerapi.SendData(streamName, inPluginId, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()
    # Obtain the inference result
    key = b"mxpi_tensorinfer0"
    keyVec = StringVector()
    keyVec.push_back(key)
    inferResult = streamManagerapi.GetProtobuf(streamName, inPluginId, keyVec)
    if inferResult.size() == 0:
        print("inferResult is null")
        exit()
    if inferResult[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
            inferResult[0].errorCode, inferResult[0].messageName.decode()))
        exit()
    # get the infer result
    inferList0 = MxpiDataType.MxpiTensorPackageList()
    inferList0.ParseFromString(inferResult[0].messageBuf)
    inferVisionData = inferList0.tensorPackageVec[0].tensorVec[0].dataStr

    output_img_data = np.frombuffer(inferVisionData, dtype=np.float32)
    output_y = colorize(output_img_data, value_min=None, value_max=None)
    output_y = output_y.reshape(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT)
    sr_img_y = Image.fromarray(np.uint8(output_y), mode="L")

    hr_img_y, _, _ = hr.convert("YCbCr").split()
    # calculate peak signal-to-noise ratio
    PSNR = calc_psnr(sr_img_y, hr_img_y)
    psnr_all.append(PSNR)
    print('PSNR: {:.2f}'.format(PSNR))


if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    # create streams by pipeline
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
                    "resizeHeight": "768",
                    "resizeWidth": "768"
                },
                "factory": "mxpi_imageresize",
                "next": "mxpi_tensorinfer0"
            },
            "mxpi_tensorinfer0": {
                "props": {
                    "dataSource": "mxpi_imagedecoder0",
                    "modelPath": "./model/VDSR_768_768.om"
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
    # test image set path
    test_image_set_path = "testSet/91-images-jpg"
    # parse command arguments
    if len(sys.argv) == 2:
        if sys.argv[1] == '':
            print('test image set path is not valid, use default config.')
        else:
            test_image_set_path = sys.argv[1]
    # check input paths
    if os.path.exists(test_image_set_path) != 1:
        print('The image set path {} does not exist.'.format(test_image_set_path))
        exit()
    # get all image files
    image_files = os.listdir(test_image_set_path)
    # sort by file name
    image_files.sort(key=lambda x: str(x[:-4]))
    print(image_files)

    # save the peak signal-to-noise ratio of each image in the test set
    psnr_all = []
    for test_image_path in image_files:
        image_file = test_image_set_path + "/" + test_image_path
        infer(image_file, streamManagerApi)
    print("average psnr = " + str(sum(psnr_all)/len(psnr_all)))
    print(psnr_all)
    # destroy streams
    streamManagerApi.DestroyAllStreams()