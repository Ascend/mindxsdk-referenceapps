#!/usr/bin/env python
# coding=utf-8

"""
Copyright(C) Huawei Technologies Co.,Ltd. 2012-2021 All rights reserved.

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

import json
import stat
import os
import cv2
import numpy as np
from preprocess import preproc

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector


OBJECT_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
               18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
               35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
               50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
               64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
               84, 85, 86, 87, 88, 89, 90]


def run_coco_eval(coco_gt_obj, image_id_list, dt_file_path):
    """
    run coco evaluation process using COCO official evaluation tool, it will print evaluation result after execution

    Args:
        coco_gt_obj: path of ground truth json file
        image_id_list: image id list
        dt_file_path: path of detected result json file

    Returns:
        None

    """
    annotation_type = 'bbox'
    print('Running test for {} results.'.format(annotation_type))
    coco_dt = coco_gt_obj.loadRes(dt_file_path)
    coco_eval = COCOeval(coco_gt_obj, coco_dt, annotation_type)
    coco_eval.params.imgIds = image_id_list
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    MODES = stat.S_IWUSR | stat.S_IRUSR
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    
    with open("../pipeline/pre_post.pipeline", 'rb') as f:
        pipelineStr = f.read()

    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    dataInput = MxDataInput()

    IMAGE_FOLDER = '../test/data/coco/val2017/'
    ANNOTATION_FILE = '../test/data/coco/annotations/instances_val2017.json'
    coco_gt = COCO(ANNOTATION_FILE)
    image_ids = coco_gt.getImgIds()
    coco_result = []

    for image_idx, image_id in enumerate(image_ids):
        image_info = coco_gt.loadImgs(image_id)[0]
        image_path = os.path.join(IMAGE_FOLDER, image_info['file_name'])
        print('Detect image: ', image_idx, ': ', image_info['file_name'], ', image id: ', image_id)
        if os.path.exists(image_path) != 1:
            print("The test image does not exist. Exit.")
            exit()
       
        with open(image_path, 'rb') as f:
            dataInput.data = f.read()
        imgs = cv2.imread(image_path)
        STREAM_NAME = b'detection'
        INPLUGIN_ID = 0
        uniqueId = streamManagerApi.SendData(STREAM_NAME, INPLUGIN_ID, dataInput)

        if uniqueId < 0:
            print("Failed to send data to stream.")
            exit()
        keys = [b"mxpi_objectpostprocessor0"]
        keyVec = StringVector()
        for key in keys:
            keyVec.push_back(key)
 

        inferResult = streamManagerApi.GetProtobuf(STREAM_NAME, 0, keyVec)


        if inferResult.size() == 0:
            print("infer_result is null")
            exit()

        if inferResult[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (
                inferResult[0].errorCode))
            exit()

        objectList = MxpiDataType.MxpiObjectList()
        objectList.ParseFromString(inferResult[0].messageBuf)

        INDS = 0 
        for results in objectList.objectVec:
            if results.classVec[0].classId == 81:
                break
            box = []
            box = {'x0': results.x0,
                   'x1': results.x1,
                    'y0': results.y0,
                    'y1': results.y1,
                      'confidence': round(results.classVec[0].confidence, 4),
                      'class': results.classVec[0].classId,
                      'text': results.classVec[0].className}
            try:
                image_result = {
                    'image_id': image_id,
                    'category_id': OBJECT_LIST[box['class']],
                    'score': float(box['confidence']),
                    'bbox': [box['x0'], box['y0'], box['x1'] - box['x0'], box['y1'] - box['y0']]
                }
                coco_result.append(image_result)
                INDS += 1
                if INDS == 100:
                    break
            except KeyError:
                print("error")
    DETECT_FILE = 'val2017_detection_result.json'
    if os.path.exists(DETECT_FILE):
        os.remove(DETECT_FILE)
    with os.fdopen(os.open(DETECT_FILE, os.O_RDWR | os.O_CREAT, MODES), 'w') as f:
        json.dump(coco_result, f, indent=4)
    run_coco_eval(coco_gt, image_ids, DETECT_FILE)

    streamManagerApi.DestroyAllStreams()
