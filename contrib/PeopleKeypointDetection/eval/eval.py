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

import json
import os
import os.path as osp
import math
from pycocotools.coco import COCO
import numpy as np


def calculate_score_w(output_path, annot_path, thr):

    with open(output_path, 'r') as f:
        output = json.load(f)

    db = COCO(annot_path)
    gt_num = len([k for k, v in db.anns.items() if v['is_valid'] == 1])
    tp_acc = 0
    fp_acc = 0
    precision = []; recall = [];
    is_matched = {}
    output_len = len(output)

    for n in range(output_len):
        image_id = output[n]['image_id']
        pred_root = output[n]['root_cam']
        score = output[n]['score']

        img = db.loadImgs(image_id)[0]
        ann_ids = db.getAnnIds(image_id)
        anns = db.loadAnns(ann_ids)
        valid_frame_num = len([item for item in anns if item['is_valid'] == 1])
        if valid_frame_num == 0:
            continue

        if str(image_id) not in is_matched:
            is_matched[str(image_id)] = [0 for _ in range(len(anns))]
        
        min_dist = 9999
        save_ann_id = -1
        for ann_id, ann in enumerate(anns):
            if ann['is_valid'] == 0:
                continue
            gt_root = np.array(ann['keypoints_cam'])
            root_idx = 14
            gt_root = gt_root[root_idx]

            dist = math.sqrt(np.sum((pred_root - gt_root) ** 2))
            if min_dist > dist:
                min_dist = dist
                save_ann_id = ann_id
        
        is_tp = False
        if save_ann_id != -1 and min_dist < thr:
            try:
                example_value = is_matched[str(image_id)][save_ann_id]
                if example_value == 0:
                    is_tp = True
                    is_matched[str(image_id)][save_ann_id] = 1
            except KeyError:
                print("data error, check ini file!")
                exit()
        
        if is_tp:
            tp_acc += 1
        else:
            fp_acc += 1
            
        precision.append(tp_acc/(tp_acc + fp_acc))
        recall.append(tp_acc/gt_num)

    AP_RESULT = 0
    for n in range(len(precision)-1):
        AP_RESULT += precision[n+1] * (recall[n+1] - recall[n])
    AP_RESULT = AP_RESULT*100
    print('AP_RESULT_root: ' + str(AP_RESULT))

if __name__ == '__main__':
    OUTPUT_PATH_W = 'bbox_root_mupots_output.json'
    ANNOT_PATH_W = 'MuPoTS_gt.json'
    THR_W = 250
    calculate_score_w(output_path_w, annot_path_w, thr_w)

