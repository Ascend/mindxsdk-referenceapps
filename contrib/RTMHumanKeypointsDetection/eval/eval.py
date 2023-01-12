#!/usr/bin/env python
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
import stat
import json
import enum

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
from proto import MxpiRTMOpenposeProtoEval_pb2  as mxpiOpenposeProto


def generate_eval_result(person_list):
    """
    Generate detect result in coco format
        
    Args:
        person_list: MxpiPersonList object, each element of which is a MxpiPersonInfo object that stores data of person

    Returns:
        None

    """
    coco_keypoints = []
    scores = []
    coor_bias = 0.5
    for person in person_list:
        skeletons = person.skeletonInfoVec
        person_score = person.score - 1 # -1 for 'neck'
        keypoints = [0] * 17 * 3
        to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        seen_idx = [1]
        # draw keypoints
        for skele in skeletons:
            part_idx1 = skele.cocoSkeletonIndex1
            part_idx2 = skele.cocoSkeletonIndex2 # two end points of a skeleton
            if part_idx1 not in seen_idx:
                seen_idx.append(part_idx1)
                center_x = skele.x0 + coor_bias
                center_y = skele.y0 + coor_bias
                keypoints[to_coco_map[part_idx1] * 3 + 0] = center_x
                keypoints[to_coco_map[part_idx1] * 3 + 1] = center_y
                keypoints[to_coco_map[part_idx1] * 3 + 2] = 1

            if part_idx2 not in seen_idx:
                seen_idx.append(part_idx2)
                center_x = skele.x1 + coor_bias
                center_y = skele.y1 + coor_bias
                keypoints[to_coco_map[part_idx2] * 3 + 0] = center_x
                keypoints[to_coco_map[part_idx2] * 3 + 1] = center_y
                keypoints[to_coco_map[part_idx2] * 3 + 2] = 1
        coco_keypoints.append(keypoints)
        scores.append(person_score)

    return coco_keypoints, scores


def run_coco_eval(gt_file_path, dt_file_path):
    """
    run coco evaluation process using COCO official evaluation tool, it will print evaluation result after execution
    
    Args:
        gt_file_path: path of ground truth json file
        dt_file_path: path of detected result json file

    Returns:
        None

    """
    annotation_type = 'keypoints'
    print('Running test for {} results.'.format(annotation_type))

    coco_gt = COCO(gt_file_path)
    coco_dt = coco_gt.loadRes(dt_file_path)

    result = COCOeval(coco_gt, coco_dt, annotation_type)
    result.evaluate()
    result.accumulate()
    result.summarize()


if __name__ == '__main__':
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("pipeline/rtmOpenposeEval.pipeline", "rb") as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    STREAM_NAME = b"Video+Openpose"
    IN_PLUGIN_ID = 0
    data_input = MxDataInput()
    IMAGE_FOLDER = 'dataset/val2017'
    ANNOTATION_FILE = 'dataset/annotations/person_keypoints_val2017.json'
    DETECT_FILE = 'val2017_keypoint_detect_result.json'
    with open(ANNOTATION_FILE, 'r') as f:
        annotations = json.load(f)
    image_list = annotations['images']
    coco_result = []
    for image_idx, image_info in enumerate(image_list):
        image_path = os.path.join(IMAGE_FOLDER, image_info['file_name'])
        image_id = image_info['id']
        print('Detect image: ', image_idx, ': ', image_info['file_name'], ', image id: ', image_id)        
        try:
            with open(image_path, 'rb') as f:
                data_input.data = f.read()
        except FileNotFoundError:
            print("Image", image_path, "doesn't exist. Exit")
            exit()
        unique_id = stream_manager_api.SendData(STREAM_NAME, IN_PLUGIN_ID, data_input)
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()
        key_vec = StringVector()
        key_vec.push_back(b"mxpi_rtmopenposepostprocesseval0")
        infer_result = stream_manager_api.GetProtobuf(STREAM_NAME, IN_PLUGIN_ID, key_vec)
        if infer_result.size() == 0:
            print("infer_result is null")
            exit()
        if infer_result[0].errorCode != 0:
            print("infer_result error. errorCode=%d" % (infer_result[0].errorCode))
            exit()
        # Get person list data
        result_person_list = mxpiOpenposeProto.MxpiPersonList()
        result_person_list.ParseFromString(infer_result[0].messageBuf)
        detect_person_list = result_person_list.personInfoVec
        eval_coco_keypoints, eval_scores = generate_eval_result(detect_person_list)
        for idx, _ in enumerate(eval_coco_keypoints):
            coco_result.append({
                'image_id': image_id,
                'category_id': 1,  # person
                'keypoints': eval_coco_keypoints[idx],
                'score': eval_scores[idx]
            })
    FLAGS = os.O_RDWR | os.O_CREAT | os.O_EXCL
    MODES = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(DETECT_FILE, FLAGS, MODES), 'w') as f:
        json.dump(coco_result, f, indent=4)
    run_coco_eval(ANNOTATION_FILE, DETECT_FILE)
    # destroy streams
    stream_manager_api.DestroyAllStreams()
