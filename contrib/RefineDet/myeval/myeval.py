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



import sys
import re
import json
import os
import stat
import random
import signal
import datetime
import threading
import time
import pickle
from functools import cmp_to_key
import cv2
import numpy as np


RESULTS_OBJECTS = []
TRUTH_SOURCE = []
TRUTH_IMAGES = []
TRUTH_ANNOTATIONS = []
APS = []
LABELS_MAP = ( 
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')



def voc_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def calc_single_object(d, class_index, current, fp, tp, ovthresh = 0.5):
    global RESULTS_OBJECTS, TRUTH_ANNOTATIONS

    image = current['image_id']
    bbox = current['bbox']
    confidence = current['score']
    gt_bboxes = [each['bbox'] for each in TRUTH_ANNOTATIONS \
                    if each['category_id'] == class_index and each['image_id'] == image]
    gt = [each for each in TRUTH_ANNOTATIONS \
                    if each['category_id'] == class_index and each['image_id'] == image]

    bbox = np.maximum(bbox, 1.)
    gt_bboxes = np.array(gt_bboxes)
    jmax = 0
    ovmax = -np.inf
    if gt_bboxes.size > 0:
        ixmin = np.maximum(gt_bboxes[:, 0], bbox[0])
        iymin = np.maximum(gt_bboxes[:, 1], bbox[1])
        ixmax = np.minimum(gt_bboxes[:, 2], bbox[2])
        iymax = np.minimum(gt_bboxes[:, 3], bbox[3])
        iw = np.maximum(ixmax - ixmin, 0.)
        ih = np.maximum(iymax - iymin, 0.)
        inters = iw * ih
        uni = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) +
               (gt_bboxes[:, 2] - gt_bboxes[:, 0]) *
               (gt_bboxes[:, 3] - gt_bboxes[:, 1]) - inters)
        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

    if ovmax > 0.5:
        tp[d] = 1. 
    else:
        fp[d] = 1.


def calc_each_class(class_index, current_class):
    global RESULTS_OBJECTS
    global TRUTH_ANNOTATIONS

    class_objects = [each for each in RESULTS_OBJECTS if each['category_id'] == class_index]
    right_objects = [each for each in TRUTH_ANNOTATIONS if each['category_id'] == class_index]

    num = len(class_objects)


    fp = np.zeros(num)
    tp = np.zeros(num)
    for i, each in enumerate(class_objects):
        calc_single_object(i, class_index, each, fp, tp)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(len(right_objects))
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)

    return ap


def mycmp(a, b):
    return b['score'] - a['score']


def main(): 
    global RESULTS_OBJECTS
    global TRUTH_SOURCE
    global TRUTH_IMAGES
    global TRUTH_ANNOTATIONS

    filename1 = './precision_analysis/result.json'
    filename2 = './precision_analysis/VOC/VOCdevkit/voc2007val.json'
    
    with open(filename1, "r") as f:
        s = f.read()
        RESULTS_OBJECTS = eval(s)
    with open(filename2, "r") as f:
        s = f.read()
        TRUTH_SOURCE = eval(s)
    TRUTH_IMAGES = TRUTH_SOURCE['images']
    TRUTH_ANNOTATIONS = TRUTH_SOURCE['annotations']

    right_results = []
    for each in RESULTS_OBJECTS:
        exist = [e for e in TRUTH_ANNOTATIONS if e['image_id'] == each['image_id']]
        if len(exist) > 0 and each['score'] >= 0.1:
            right_results.append(each)

    right_results.sort(key = cmp_to_key(lambda A, B: B['score'] - A['score']))
    RESULTS_OBJECTS = right_results

    for i, each_class in enumerate(LABELS_MAP):
        ap = calc_each_class(i+1, each_class)
        APS.append(ap)

    print("mAP:", sum(APS)/len(APS))


if __name__ == '__main__':
    main()