# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import os
import argparse
from glob import glob
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from PIL import ImageDraw

from include.box_utils import decode, decode_landm
from include.prior_box import PriorBox
from include.py_cpu_nms import py_cpu_nms
cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}


def preprocess_for_main(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        target_size = 1000
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = target_size / im_size_max
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)
        width_pad = target_size - img.shape[1]
        left = 0 
        right = width_pad
        height_pad = target_size - img.shape[0]
        top = 0
        bottom = height_pad
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        im_height, im_width, _ = img.shape
        img = img.flatten()
        return img , [resize, left, top, right, bottom]
        

def preprocess(image_path):
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)
        # testing scale
        target_size = 1000
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = target_size / im_size_max
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)
        width_pad = target_size - img.shape[1]
        left = 0 
        right = width_pad
        height_pad = target_size - img.shape[0]
        top = 0
        bottom = height_pad
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img = torch.from_numpy(img).unsqueeze(0).byte()
        info = np.array(resize, dtype=np.float32)
        return img.numpy() , info


def postprocess(loc , conf , landms , resize):
        loc = np.reshape(loc, [1, 41236, 4])
        conf = np.reshape(conf ,  [1, 41236, 2])
        landms = np.reshape(landms , [1, 41236, 10])
        scale = torch.ones(4,).fill_(1000)
        loc = torch.Tensor(loc)
        conf = torch.Tensor(conf)
        landms = torch.Tensor(landms)
        priorbox = PriorBox(cfg_mnet, image_size=(1000, 1000))
        priors = priorbox.forward()
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, [0.1, 0.2])
        boxes = boxes * scale / resize
        boxes = boxes.numpy()
        scores = conf.squeeze(0).data.numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, [0.1, 0.2])
        scale1 = torch.ones(10,).fill_(1000)

        landms = landms * scale1 / resize
        landms = landms.numpy()

        inds = np.where(scores > 0.02)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.4)
        dets = dets[keep, :]
        landms = landms[keep]

        dets = np.concatenate((dets, landms), axis=1)
        bboxs = dets
        bboxs_num = str(len(bboxs)) + "\n"
        a = Image.open("test.jpg")
        result = ''
        count = 0
        for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                if confidence != 0:
                    count += 1
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                result += line
                aa = ImageDraw.ImageDraw(a)
                aa.rectangle((x, y, x+w, y+h), outline='red', width=3)
        return result , count
