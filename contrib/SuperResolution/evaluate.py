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


def infer(input_image_path, stream_manager):
    """
    image super-resolution inference
    :param input_image_path: input image path
    :param stream_manager: stream_manager
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
    data_input = MxDataInput()
    data_input.data = input_image_data

    # inputs data to a specified stream based on stream_name
    stream_name = b'superResolution'
    in_plugin_id = 0
    unique_id = stream_manager.SendData(stream_name, in_plugin_id, data_input)
    if unique_id < 0:
        print("Failed to send data to stream.")
        exit()
    # Obtain the inference result
    key = b"mxpi_tensorinfer0"
    key_vec = StringVector()
    key_vec.push_back(key)
    infer_result = stream_manager.GetProtobuf(stream_name, in_plugin_id, key_vec)
    if infer_result.size() == 0:
        print("infer_result is null")
        exit()
    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
            infer_result[0].errorCode, infer_result[0].messageName.decode()))
        exit()
    # get the infer result
    infer_list = MxpiDataType.MxpiTensorPackageList()
    infer_list.ParseFromString(infer_result[0].messageBuf)
    vision_data = infer_list.tensorPackageVec[0].tensorVec[0].dataStr

    output_img_data = np.frombuffer(vision_data, dtype=np.float32)
    output_y = colorize(output_img_data, value_min=None, value_max=None)
    output_y = output_y.reshape(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT)
    sr_img_y = Image.fromarray(np.uint8(output_y), mode="L")

    hr_img_y, _, _ = hr.convert("YCbCr").split()
    # calculate peak signal-to-noise ratio
    psnr = calc_psnr(sr_img_y, hr_img_y)
    psnr_all.append(psnr)
    print('PSNR: {:.2f}'.format(psnr))


if __name__ == '__main__':
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
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
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    # test image set path
    TEST_IMAGE_SET_PATH = "testSet/91-images-jpg"
    # parse command arguments
    if len(sys.argv) == 2:
        if sys.argv[1] == '':
            print('test image set path is not valid, use default config.')
        else:
            TEST_IMAGE_SET_PATH = sys.argv[1]
    # check input paths
    if os.path.exists(TEST_IMAGE_SET_PATH) != 1:
        print('The image set path {} does not exist.'.format(TEST_IMAGE_SET_PATH))
        exit()
    # get all image files
    image_files = os.listdir(TEST_IMAGE_SET_PATH)
    # sort by file name
    image_files.sort(key=lambda x: str(x[:-4]))
    print(image_files)

    # save the peak signal-to-noise ratio of each image in the test set
    psnr_all = []
    for test_image_path in image_files:
        image_file = TEST_IMAGE_SET_PATH + "/" + test_image_path
        infer(image_file, stream_manager_api)
    print("average psnr = " + str(sum(psnr_all)/len(psnr_all)))
    print(psnr_all)
    # destroy streams
    stream_manager_api.DestroyAllStreams()