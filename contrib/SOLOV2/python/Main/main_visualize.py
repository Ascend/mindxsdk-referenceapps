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
import cv2
import numpy as np
import argparse
import mmcv
import pycocotools.mask as mask_util
from scipy import ndimage
from mmdet.datasets import build_dataset

def get_masks(result, num_classes=80):
    for cur_result in result:
        masks = [[] for _ in range(num_classes)]
        if cur_result is None:
            return masks
        seg_pred = cur_result[0].astype(np.uint8)
        cate_label = cur_result[1].astype(np.int)
        cate_score = cur_result[2].astype(np.float)
        num_ins = seg_pred.shape[0]  # 100
        for idx in range(num_ins):
            cur_mask = seg_pred[idx, ...]
            rle = mask_util.encode(
                np.array(cur_mask[:, :, np.newaxis], order='F'))[0]
            # print("rle", rle)
            rst = (rle, cate_score[idx])
            masks[cate_label[idx]].append(rst)
        return masks

file = "../coco/val2017/000000281447.jpg"
parser = argparse.ArgumentParser(description='model')
parser.add_argument('--dataset_path', default="../coco/val2017/")
parser.add_argument('--anno_path', default='../coco/annotations/instances_val2017.json')
parser.add_argument('--model_config', default="SOLO/configs/solov2/solov2_r50_fpn_8gpu_1x.py")
parser.add_argument("--model_input_height", default=800, type=int, help='input tensor height')
parser.add_argument("--model_input_width", default=1216, type=int, help='input tensor width')
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

    pipelineStr = json.dumps(pipeline).encode()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    dataInput = MxDataInput()
    if os.path.exists(file) != 1:
        print("The test image does not exist.")
    with open(file, 'rb') as f:
        dataInput.data = f.read()
    img = cv2.imread(file)

    streamName = b'detection'
    inPluginId = 0
    uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    keys = [b"mxpi_objectpostprocessor0"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)

    infer_result = streamManagerApi.GetProtobuf(streamName, 0, keyVec)

    if infer_result.size() == 0:
        print("infer_result is null")
        exit()

    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" % (
            infer_result[0].errorCode))
        exit()

    # get infer result
    objectList = MxpiDataType.MxpiObjectList()
    objectList.ParseFromString(infer_result[0].messageBuf)
    seg_show = img.copy()
    num_mask = len(objectList.objectVec)
    for idx in range(num_mask):
        idx = -(idx + 1)
        result = objectList.objectVec[idx]
        if result.classVec[0].classId == 81:
            print("no result!")
            cv2.imwrite("./result.jpg", img)
            break
        cur_mask = result.imageMask.dataStr
        ori_h = result.imageMask.shape[0]
        ori_w = result.imageMask.shape[1]
        cur_mask = np.frombuffer(cur_mask, dtype=np.uint8).reshape(ori_h, ori_w)
        if cur_mask.sum() == 0:
            continue
        color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        cur_mask_bool = cur_mask.astype(np.bool)
        seg_show[cur_mask_bool] = img[cur_mask_bool] * 0.5 + color_mask * 0.5
        cur_cate = result.classVec[0].classId
        cur_score = round(result.classVec[0].confidence, 4)

        label_text = result.classVec[0].className
        # label_text += '|{:.02f}'.format(cur_score)
        # center
        center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
        vis_pos = (max(int(center_x) - 10, 0), int(center_y))
        cv2.putText(seg_show, label_text, vis_pos,
                    cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))  # green
    mmcv.imwrite(seg_show, './result.jpg')

    # destroy streams
    streamManagerApi.DestroyAllStreams()