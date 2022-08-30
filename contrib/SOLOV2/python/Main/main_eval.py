#!/usr/bin/env python
# coding=utf-8

"""
Copyright(C) Huawei Technologies Co.,Ltd. 2012-2022 All rights reserved.

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

import os
import argparse
import cv2
import numpy as np
import mmcv
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from mmdet.core import coco_eval, results2json, results2json_segm
from mmdet.datasets import build_dataset
from tqdm import tqdm

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import MxDataInput, InProtobufVector, MxProtobufIn, StringVector, StreamManagerApi


def get_case_masks(result_, num_classes_=80):
    for r in result_:
        masks = [[] for _ in range(num_classes_)]
        if r is None:
            return masks
        seg = r[0].astype(np.uint8)
        label = r[1].astype(np.int)
        score = r[2].astype(np.float)
        ans_num = seg.shape[0]  # 100
        for idx in range(ans_num):
            current = seg[idx, ...]
            enc = mask_util.encode(
                np.array(current[:, :, np.newaxis], order='F'))[0]
            rst = (enc, score[idx])
            masks[label[idx]].append(rst)
        return masks

parser = argparse.ArgumentParser(description='model')
parser.add_argument('--dataset_path', default="../coco/val2017/")
parser.add_argument('--anno_path', default='../coco/annotations/instances_val2017.json')
parser.add_argument('--model_config', default="./SOLOV2/SOLO/configs/solov2/solov2_r50_fpn_8gpu_1x.py")
args = parser.parse_args()


if __name__ == '__main__':
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # build dataset
    cfg = mmcv.Config.fromfile(args.model_config)
    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = args.anno_path
    cfg.data.test.img_prefix = args.dataset_path
    dataset = build_dataset(cfg.data.test)
    num_classes = len(dataset.CLASSES)

    with open("../pipeline/solov2.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    dataInput = MxDataInput()
    results = []
    coco_gt = COCO(args.anno_path)
    image_ids = coco_gt.getImgIds()
    for image_idx, image_id in tqdm(enumerate(image_ids)):
        image_info = coco_gt.loadImgs(image_id)[0]
        image_path = os.path.join(args.dataset_path, image_info['file_name'])
        print('Detect image: ', image_idx, ': ', image_info['file_name'], ', image id: ', image_id)
        if os.path.exists(image_path) != 1:
            print("The test image does not exist. Exit.")
            exit()
        with open(image_path, 'rb') as f:
            dataInput.data = f.read()

        img = cv2.imread(image_path)
        uniqueId = streamManagerApi.SendData(b'detection', 0, dataInput)
        if uniqueId < 0:
            print("Failed to send data to stream.")
            exit()

        keys = [b"mxpi_objectpostprocessor0"]
        keyVec = StringVector()
        for key in keys:
            keyVec.push_back(key)

        infer_result = streamManagerApi.GetProtobuf(b'detection', 0, keyVec)

        if infer_result.size() == 0:
            print("infer_result is null")
            exit()

        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
            exit()

        # get infer result
        objectList = MxpiDataType.MxpiObjectList()
        objectList.ParseFromString(infer_result[0].messageBuf)
        seg_show = img.copy()
        num_mask = len(objectList.objectVec)

        r_seg = []
        r_label = []
        r_score = []
        for item in objectList.objectVec:
            cur_mask = item.imageMask.dataStr
            ori_h = item.imageMask.shape[0]
            ori_w = item.imageMask.shape[1]
            cur_mask = np.frombuffer(cur_mask, dtype=np.uint8).reshape(ori_h, ori_w)
            cur_cate = item.classVec[0].classId
            cur_score = item.classVec[0].confidence
            r_seg.append(cur_mask)
            r_label.append(cur_cate)
            r_score.append(cur_score)
        r_seg = np.array(r_seg)
        r_label = np.array(r_label)
        r_score = np.array(r_score)
        result = [r_seg, r_label, r_score]
        result = get_case_masks([result], num_classes)
        results.append(result)

    result_files = results2json_segm(dataset, results, "results_solo.pkl")
    coco_eval(result_files, ["segm"], dataset.coco)

    # destroy streams
    streamManagerApi.DestroyAllStreams()