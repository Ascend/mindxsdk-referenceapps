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
import time
from utils import colorize, calc_psnr
import numpy as np
from PIL import Image
from mindx.sdk import base
from mindx.sdk.base import Size, Rect
from mindx.sdk.base import ImageProcessor 
DEFAULT_IMAGE_WIDTH = 768
DEFAULT_IMAGE_HEIGHT = 768
SCALE = 3
FONT_SIZE = 16
OFFSET_5 = 5
OFFSET_20 = 20


def infer(input_image_path,imageProcessor1,imageProcessor2,model_):
    """
	  image super-resolution inference
	  :param input_image_path: input image path
	  :param imageProcessor1: imageProcessor object to decode
 	  :param imageProcessor1: imageProcessor object to resize
    :param model_: model object to infer    
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
    image_path = "./ilr_image.jpg"
    ilr.save(image_path, format='JPEG')

    # V2 decode and resize
    decodedImg = imageProcessor1.decode(image_path, base.nv12)
    size_cof = Size(DEFAULT_IMAGE_WIDTH,DEFAULT_IMAGE_HEIGHT)寸
    resizedImg = imageProcessor2.resize(decodedImg, size_cof, base.huaweiu_high_order_filter)
    # V2 infer
    imgTensor = [resizedImg.to_tensor()]
    outputs = model_.infer(imgTensor)
    # get the infer result
    output0 = outputs[0]
    output0.to_host()
    output_img_data = np.array(output0)

    output_y = colorize(output_img_data, value_min=None, value_max=None)
    output_y = output_y.reshape(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT)
    sr_img_y = Image.fromarray(np.uint8(output_y), mode="L")

    hr_img_y, _, _ = hr.convert("YCbCr").split()
    # calculate peak signal-to-noise ratio
    PSNR = calc_psnr(sr_img_y, hr_img_y)
    psnr_all.append(PSNR)
    print('PSNR: {:.2f}'.format(PSNR))


if __name__ == '__main__':
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
    #V2 initialize
    base.mx_init()
    device_id = 0
    imageProcessor1 = ImageProcessor(device_id)
    imageProcessor2 = ImageProcessor(device_id)
    model_path = "model/VDSR_768_768.om"
    model_ = base.model(model_path, deviceId=device_id)
    # save the peak signal-to-noise ratio of each image in the test set
    psnr_all = []
    start=time.time()
    # infer
    for test_image_path in image_files:
        image_file = test_image_set_path + "/" + test_image_path
        infer(image_file,imageProcessor1,imageProcessor2,model_)
    print("average psnr = " + str(sum(psnr_all)/len(psnr_all)))
    print(psnr_all)
    end=time.time()
    print('Running time: %s Seconds'%(end-start))
 