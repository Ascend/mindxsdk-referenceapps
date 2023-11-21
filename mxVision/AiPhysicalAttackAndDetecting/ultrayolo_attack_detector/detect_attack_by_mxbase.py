#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
Description: Main process of attack detect.
Author: MindX SDK
Create: 2023
History: NA
"""

import os
import numpy as np
import cv2
import yaml
import torch
from mindx.sdk.base import Tensor, Model
from ultrayolo_attack_detector.utils import Annotator, colors, non_max_suppression, scale_coords, letterbox


class DetectObjWithYolov3OMUltra(object):
    def __init__(self, device_id=0):
        self.model_path = "ultrayolo_attack_detector/model/ultra_best.om"
        self.label_names_path = "ultrayolo_attack_detector/model/coco.yaml"
        self.check()
        self.device_id = device_id
        self.model_ = Model(self.model_path, deviceId=device_id)  # 创造模型对象
        print("finished initialization.")
    
    def check(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("attack detect model_path not found.")
        if not os.path.exists(self.label_names_path):
            raise FileNotFoundError("attack detect label_names_path not found.")

    def read_names(self):
        with open(self.label_names_path, 'r', encoding='utf-8') as file:
            coco_names = yaml.safe_load(stream=file)
        return coco_names.get('names')

    def pre_process_with_cv2(self, frame):
        img = letterbox(frame, 640, stride=32, auto=False)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = img.astype(np.float32)
        im = np.ascontiguousarray(img)
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        return im, frame  # origin_img_h, origin_img_w

    def post_process(self, pred, pre_processed_img, save_path, original_img, save_img=False, hide_labels=False,
                     hide_conf=False):
        found_patch = False
        class_names = self.read_names()
        # NMS
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            im0 = original_img.copy()

            annotator = Annotator(im0, line_width=5, example=str(class_names))
            if len(det):
                found_patch = True
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(pre_processed_img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    #if save_img:  # or save_crop: #or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (
                        class_names[c] if hide_conf else f'{class_names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)
        return im0, found_patch

    def detect_image_with_yolov3_om(self, frame):
        # detecting objects:
        pre_processed_img, original_img = self.pre_process_with_cv2(frame)

        resizedImg = Tensor(pre_processed_img)
        resizedImg.to_device(self.device_id)
        imgTensor = [resizedImg]  # 将Image类转为Tensor
        results = self.model_.infer(imgTensor)  # 使用模型对Tensor对象进行推理
        results[0].to_host()
        pred = np.array(results[0], dtype=object).astype(np.float32)
        pred = torch.from_numpy(pred)

        save_path = "got_attack.jpg"
        frame_with_bbox, found_patch = self.post_process(pred, pre_processed_img, save_path, original_img)

        return frame_with_bbox, found_patch
