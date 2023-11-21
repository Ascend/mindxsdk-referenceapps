#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
Description: Main process of object detect.
Author: MindX SDK
Create: 2023
History: NA
"""

import os
import numpy as np
import cv2
from mindx.sdk.base import Tensor, Rect
from mindx.sdk.base import Image, Model, ImageProcessor, Size
from darknet_ob_detector.utils import load_class_names, post_processing, plot_boxes_cv2


class DetectObjWithYolov3OM(object):
    def __init__(self,device_id=0):
        self.model_path = "darknet_ob_detector/model/darknet_aarch64_Ascend310B1_input.om"
        self.label_names_path = "darknet_ob_detector/model/coco.names"
        self.check()
        self.device_id = device_id 
        self.model_ = Model(self.model_path, deviceId=device_id) # 创造模型对象
        self.model_width = 416
        self.model_height = 416
        self.class_names = load_class_names(self.label_names_path)
        print("finished initialization.")
    
    def check(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("obj detect model_path not found.")
        if not os.path.exists(self.label_names_path):
            raise FileNotFoundError("obj detect label_names_path not found.")

    def pre_process_with_cv2(self, frame):
        origin_img_w, origin_img_h = frame.shape[1], frame.shape[0]
        resized = cv2.resize(frame, (self.model_width, self.model_height), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32).copy()
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0
        return img_in, origin_img_h, origin_img_w

    def post_process_cv2(self,resized_image, infer_output, frame, count):
        boxes = post_processing(resized_image, 0.8, 0.6, infer_output)    
        frame, found_person = plot_boxes_cv2(frame, boxes[0], savename=None, class_names=self.class_names)
        return frame, found_person

    def detect_image_with_yolov3_om(self,frame):
        img_in, origin_img_h, origin_img_w = self.pre_process_with_cv2(frame)
        resizedImg = Tensor(img_in)
        resizedImg.to_device(self.device_id)
        imgTensor = [resizedImg] # 将Image类转为Tensor
        results = self.model_.infer(imgTensor) # 使用模型对Tensor对象进行推理
        results[0].to_host()
        results[1].to_host()
        outputs = [np.array(results[0]), np.array(results[1])]
        frame_with_bbox, found_person = self.post_process_cv2(resizedImg, outputs, frame, count=0)
        return frame_with_bbox, found_person
