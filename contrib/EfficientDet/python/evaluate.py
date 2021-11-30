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

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--pipeline', type=str, default='pipeline/EfficientDet-d0.pipeline', help='pipeline of different models used to evaluate')
ap.add_argument('-o', '--output', type=str, default='val2017_detection_result.json', help='name of detection result json file')
args = ap.parse_args()


def run_coco_eval(coco_gt, image_ids, dt_file_path):
    """
    run coco evaluation process using COCO official evaluation tool, it will print evaluation result after execution

    Args:
        coco_gt: path of ground truth json file
        image_ids: image id list
        dt_file_path: path of detected result json file

    Returns:
        None

    """
    annotation_type = 'bbox'
    print('Running test for {} results.'.format(annotation_type))
    coco_dt = coco_gt.loadRes(dt_file_path)
    coco_eval = COCOeval(coco_gt, coco_dt, annotation_type)
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    pipeline_path = args.pipeline
    print('pipeline path: ', pipeline_path)
    with open(pipeline_path, "rb") as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    stream_name = b"detection"
    in_plugin_id = 0
    data_input = MxDataInput()
    image_folder = 'dataset/val2017'
    annotation_file = 'dataset/annotations/instances_val2017.json'
    detect_file = args.output
    print('output detection json file path: ', detect_file)
    coco_gt = COCO(annotation_file)
    image_ids = coco_gt.getImgIds()
    print('Test on coco2017 test-dev dataset, ', len(image_ids), ' images in total.')
    coco_result = []
    for image_idx, image_id in enumerate(image_ids):
        image_info = coco_gt.loadImgs(image_id)[0]
        image_path = os.path.join(image_folder, image_info['file_name'])
        print('Detect image: ', image_idx, ': ', image_info['file_name'], ', image id: ', image_id)
        if os.path.exists(image_path) != 1:
            print("The test image does not exist. Exit.")
            exit()
        with open(image_path, 'rb') as f:
            data_input.data = f.read()
        unique_id = stream_manager_api.SendData(stream_name, in_plugin_id, data_input)
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()
        key_vec = StringVector()
        key_vec.push_back(b"mxpi_objectpostprocessor0")
        infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
        if infer_result.size() == 0:
            print("infer_result is null")
            continue
        if infer_result[0].errorCode != 0:
            print("infer_result error. errorCode=%d" % (infer_result[0].errorCode))
            exit()
        # Get bounding box list
        objectList = MxpiDataType.MxpiObjectList()
        objectList.ParseFromString(infer_result[0].messageBuf)
        for obj in objectList.objectVec:
            box = {'x0': obj.x0,
                   'x1': obj.x1,
                   'y0': obj.y0,
                   'y1': obj.y1,
                   'class': obj.classVec[0].classId,
                   'confidence': obj.classVec[0].confidence}
            # box: x0, y0, w, h
            image_result = {
                'image_id': image_id,
                'category_id': box['class'] + 1,
                'score': float(box['confidence']),
                'bbox': [box['x0'], box['y0'], box['x1'] - box['x0'], box['y1'] - box['y0']]
            }
            coco_result.append(image_result)

    if os.path.exists(detect_file):
        os.remove(detect_file)
    with open(detect_file, 'w') as f:
        json.dump(coco_result, f, indent=4)
    run_coco_eval(coco_gt, image_ids, detect_file)

    # destroy streams
    stream_manager_api.DestroyAllStreams()

