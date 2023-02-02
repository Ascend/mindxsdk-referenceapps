# !/usr/bin/env python
# coding=utf-8

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
import sys
import os
import argparse
import json
import cv2
import numpy as np
import time
import mmcv

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, StringVector, MxProtobufIn
from mmdet.core import bbox2result
from mmdet.core import encode_mask_results
from mmdet.datasets import CocoDataset


def resizeImage(img, size):
    old_h = img.shape[0]
    old_w = img.shape[1]
    scale_ratio = min(size[0] / old_w, size[1] / old_h)
    new_w = int(np.floor(old_w * scale_ratio))
    new_h = int(np.floor(old_h * scale_ratio))
    img = mmcv.imresize(img, (new_w, new_h), backend='cv2')
    return img, scale_ratio


def preprocess(imagePath):
    image = mmcv.imread(imagePath, backend='cv2')
    image, SCALE = resizeImage(image, (1333, 800))
    h = image.shape[0]
    w = image.shape[1]
    mean = np.array([102.9801, 115.9465, 122.7717], dtype=np.float32)
    std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    norm_img_data = mmcv.imnormalize(image, mean, std, to_rgb=False)
    pad_left = (1333 - w) // 2
    pad_top = (800 - h) // 2
    pad_right = 1333 - w - pad_left
    pad_bottom = 800 - h - pad_top
    image_for_infer = mmcv.impad(norm_img_data,
                                 padding=(pad_left, pad_top, pad_right,
                                          pad_bottom),
                                 pad_val=0)
    image_for_infer = image_for_infer.transpose(2, 0, 1)
    return image_for_infer, [SCALE, pad_left, pad_top, pad_right, pad_bottom]


if __name__ == '__main__':
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    pipeline_path = './pipeline/FCOSdetection.pipeline'
    print('pipeline path: ', pipeline_path)
    with open(pipeline_path, "rb") as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    stream_name = b"detection"
    in_plugin_id = 0

    image_folder = './dataset/val2017'
    annotation_file = './dataset/annotations/instances_val2017.json'

    coco_gt = COCO(annotation_file)
    image_ids = coco_gt.getImgIds()
    print('Test on coco2017 test-dev dataset, ', len(image_ids),
          ' images in total.')

    coco_result = []

    for image_idx, image_id in enumerate(image_ids):
        print('image_idx = %d image_id = %d.' %
              (image_idx, image_id))  # image_id is name of picture
        image_info = coco_gt.loadImgs(image_id)[0]  # 获取这张测试图片的信息
        image_path = os.path.join(image_folder,
                                  image_info['file_name'])  # 获取测试图片的路径
        print('Detect image: ', image_idx, ': ', image_info['file_name'],
              ', image id: ', image_id)
        if os.path.exists(image_path) != 1:
            print("The test image does not exist. Exit.")
            exit()

        tensorData, return_image = preprocess(image_path)
        tensor = tensorData[None, :]

        visionList = MxpiDataType.MxpiVisionList()
        visionVec = visionList.visionVec.add()
        visionInfo = visionVec.visionInfo

        visionInfo.width = 1333
        visionInfo.height = 800
        visionInfo.widthAligned = 1333
        visionInfo.heightAligned = 800
        visionData = visionVec.visionData
        visionData.dataStr = tensor.tobytes()
        visionData.deviceId = 0
        visionData.memType = 0
        visionData.dataSize = len(tensor)

        # stream_name = b"detection"
        KEYVALUE = b'appsrc0'
        protobufVec = InProtobufVector()
        protobuf = MxProtobufIn()
        protobuf.key = KEYVALUE
        protobuf.type = b"MxTools.MxpiVisionList"
        protobuf.protobuf = visionList.SerializeToString()
        protobufVec.push_back(protobuf)
        unique_id = stream_manager_api.SendProtobuf(stream_name, in_plugin_id,
                                                    protobufVec)

        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()
        key_vec = StringVector()
        key_vec.push_back(b"mxpi_objectpostprocessor0")
        infer_result = stream_manager_api.GetProtobuf(
            stream_name, 0, key_vec)  # 获取本张图片的测试结果（目标框）
        if infer_result.size() == 0:
            print("infer_result is null")
            continue
        if infer_result[0].errorCode != 0:
            print("infer_result error. errorCode=%d" %
                  (infer_result[0].errorCode))
            exit()

        objectList = MxpiDataType.MxpiObjectList()
        objectList.ParseFromString(infer_result[0].messageBuf)
        SCALE, pad_left, pad_top, pad_right, pad_bottom = return_image  # 用来复原目标框
        bboxfileName = 'bboxres' + str(image_id) + '.bin'
        classfileName = 'class' + str(image_id) + '.bin'
        bboxresultList = []
        classresultList = []
        for obj in objectList.objectVec:
            box = {
                'x0': max(int((obj.x0 - pad_left) / SCALE), 0),
                'x1': max(int((obj.x1 - pad_left) / SCALE), 0),
                'y0': max(int((obj.y0 - pad_top) / SCALE), 0),
                'y1': max(int((obj.y1 - pad_top) / SCALE), 0),
                'class': obj.classVec[0].classId,
                'confidence': obj.classVec[0].confidence
            }
            # box: x0, y0, w, h
            bboxresultList.append(max(float((obj.x0 - pad_left) / SCALE), 0.0))
            bboxresultList.append(max(float((obj.y0 - pad_top) / SCALE), 0.0))
            bboxresultList.append(max(float((obj.x1 - pad_left) / SCALE), 0.0))
            bboxresultList.append(max(float((obj.y1 - pad_top) / SCALE), 0.0))
            bboxresultList.append(obj.classVec[0].confidence)
            classresultList.append(obj.classVec[0].classId)

            bboxresultnumpy = np.array(bboxresultList, dtype=np.float32)
            classresultnumpy = np.array(classresultList, dtype=np.int64)
            bboxresultnumpy.tofile('./binresult/' + bboxfileName)
            classresultnumpy.tofile('./binresult/' + classfileName)

    coco_dataset = CocoDataset(
        ann_file='./dataset/annotations/instances_val2017.json', pipeline=[])
    results = []

    for ids in coco_dataset.img_ids:
        print('image ids = {}'.format(ids))
        bbox_results = []
        # read bbox information
        BBOX_RES_PATH = './binresult/bboxres' + str(ids) + '.bin'
        REBLES_RES_PATH = './binresult/class' + str(ids) + '.bin'
        bboxes = np.fromfile(BBOX_RES_PATH, dtype=np.float32)
        bboxes = np.reshape(bboxes, [100, 5])
        bboxes.tolist()
        # read label information
        labels = np.fromfile(REBLES_RES_PATH, dtype=np.int64)
        labels = np.reshape(labels, [100, 1])
        labels.tolist()
        bbox_results = [bbox2result(bboxes, labels[:, 0], 80)]

        result = bbox_results
        results.extend(result)
    print('Evaluating...')
    eval_results = coco_dataset.evaluate(results,
                                         metric=[
                                             'bbox',
                                         ],
                                         classwise=True)
    print(eval_results)