#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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

import io
import json
import os
import sys
import time
from utils import colorize, calc_psnr
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from mindx.sdk import base
from mindx.sdk.base import Size, Rect
from mindx.sdk.base import ImageProcessor 
DEFAULT_IMAGE_WIDTH = 768
DEFAULT_IMAGE_HEIGHT = 768
SCALE = 3
FONT_SIZE = 16
OFFSET_5 = 5
OFFSET_20 = 20
MIN_IMAGE_SIZE = 32
MAX_IMAGE_SIZE = 8192


if __name__ == '__main__':
    # input image path
    INPUT_IMAGE_PATH = "./image/head.jpg"
    # parse command arguments
    if len(sys.argv) == 2:
        if sys.argv[1] == '':
            print('input image path is not valid, use default config.')
        else:
            INPUT_IMAGE_PATH = sys.argv[1]

    # check input image
    if os.path.exists(INPUT_IMAGE_PATH) != 1:
        print('The {} does not exist.'.format(INPUT_IMAGE_PATH))
        exit()
    else:
        image = Image.open(INPUT_IMAGE_PATH)
        if image.format != 'JPEG':
            print('input image only support jpg, curr format is {}.'.format(image.format))
            exit()
        elif image.width < MIN_IMAGE_SIZE or image.width > MAX_IMAGE_SIZE:
            print('input image width must in range [{}, {}], curr width is {}.'.format(
                MIN_IMAGE_SIZE, MAX_IMAGE_SIZE, image.width))
            exit()
        elif image.height < MIN_IMAGE_SIZE or image.height > MAX_IMAGE_SIZE:
            print('input image height must in range [{}, {}], curr height is {}.'.format(
                MIN_IMAGE_SIZE, MAX_IMAGE_SIZE, image.height))
            exit()

    image = image.convert('RGB')
    # high resolution image, (that is ground true)
    hr = image.resize((DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT), resample=Image.BICUBIC)
    # low resolution image
    lr = hr.resize((hr.width // SCALE, hr.height // SCALE), resample=Image.BILINEAR)
    # interpolated low-resolution image
    ilr = lr.resize((lr.width * SCALE, lr.height * SCALE), resample=Image.BICUBIC)
    IMAGE_PATH = "./ilr_image.jpg"
    ilr.save(IMAGE_PATH, format='JPEG')
    
    start = time.time()
    # V2 initialize
    DEVICE_ID = 0  
    base.mx_init()
    # V2 decode and resize
    imageProcessor1 = ImageProcessor(DEVICE_ID) 
    decodedImg = imageProcessor1.decode(IMAGE_PATH, base.nv12)  
    imageProcessor2 = ImageProcessor(DEVICE_ID)  
    size_cof = Size(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT)  
    resizedImg = imageProcessor2.resize(decodedImg, size_cof, base.huaweiu_high_order_filter)  
    # V2 infer
    imgTensor = [resizedImg.to_tensor()]  
    MODEL_PATH = "model/VDSR_768_768.om"  
    model_ = base.model(MODEL_PATH, deviceId=DEVICE_ID) 
    outputs = model_.infer(imgTensor)   
    
    end = time.time()
    print('V2 Running time: %s Seconds.' % (end - start))
    
    # get the infer result 
    output0 = outputs[0]  
    output0.to_host()  
    output_img_data = np.array(output0)  
    output_y = colorize(output_img_data, value_min=None, value_max=None)
    output_y = output_y.reshape(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT)
    sr_img_y = Image.fromarray(np.uint8(output_y), mode="L")

    # construct super-resolution images
    hr_YCbCr = hr.convert("YCbCr")
    hr_img_y, cb, cr = hr_YCbCr.split()
    output_img = Image.merge("YCbCr", [sr_img_y, cb, cr]).convert("RGB")
    output_img.save("./super_resolution_image.jpg", format='JPEG')

    # calculate peak signal-to-noise ratio
    PSNR = calc_psnr(sr_img_y, hr_img_y)
    print('PSNR: {:.2f}'.format(PSNR))

    # create a canvas for visualization
    target = Image.new('RGB', (lr.width + OFFSET_5 + output_img.width, output_img.height), "white")
    # splice the pictures line by line
    target.paste(lr, (0, output_img.height - lr.height))
    target.paste(output_img, (lr.width + OFFSET_5, 0))
 
    # create a brush to write text to the picture
    draw = ImageDraw.Draw(target)
    # set font type and size
    TYPE = "./font/SourceHanSansCN-Normal-2.otf"
    SIZE = FONT_SIZE
    COLOR = (0, 0, 0)
    PSNR_CONTENT = 'PSNR: {:.2f}'.format(PSNR)
    psnr_location = (0, output_img.height - lr.height - OFFSET_20)
    font = ImageFont.truetype(TYPE, SIZE)
    # draw into the picture according to the position, content, color and font
    draw.text(psnr_location, PSNR_CONTENT, COLOR, font)
    # save visualization results
    _, fileName = os.path.split(INPUT_IMAGE_PATH)
    out_path = "./V2result/" + fileName
    target.save(out_path)