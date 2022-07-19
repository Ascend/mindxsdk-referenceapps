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


if __name__ == '__main__':
    OUTPUT_PATH_W = 'bbox_root_mupots_output.json'
    ANNOT_PATH_W = 'MuPoTS_gt.json'
    with open(OUTPUT_PATH_W, 'r') as f:
        output = json.load(f)

    dbcoco = COCO(ANNOT_PATH_W)
    gt_num = len([k for k, v in dbcoco.anns.items() if v['is_valid'] == 1])
    tp_ac = 0
    fp_ac = 0
    pre = [];
    rec = [];
    isare_matched = {}
    output_len = len(output)

    for n in range(output_len):
        image_idin = output[n]['image_id']
        pred_inroot = output[n]['root_cam']

        img = dbcoco.loadImgs(image_idin)[0]
        ann_ids = dbcoco.getAnnIds(image_idin)
        anns = dbcoco.loadAnns(ann_ids)
        valid_frame_num_count = len([item for item in anns if item['is_valid'] == 1])
        if valid_frame_num_count == 0:
            continue

        if str(image_idin) not in isare_matched:
            isare_matched[str(image_idin)] = [0 for _ in range(len(anns))]

        min_dist_d = 9999
        save_ann_id_num = -1
        for ann_id, ann in enumerate(anns):
            if ann['is_valid'] == 0:
                continue
            gt_root = np.array(ann['keypoints_cam'])
            mid_root_idx = 14
            gt_root = gt_root[mid_root_idx]

            dist_re = math.sqrt(np.sum((pred_inroot - gt_root) ** 2))
            if min_dist_d > dist_re:
                min_dist_d = dist_re
                save_ann_id_num = ann_id

        THR_W = 250
        is_tp = False
        if save_ann_id_num != -1 and min_dist_d < THR_W:
            try:
                example_value = isare_matched[str(image_idin)][save_ann_id_num]
                if example_value == 0:
                    is_tp = True
                    isare_matched[str(image_idin)][save_ann_id_num] = 1
            except KeyError:
                print("data error, check ini file!")
                exit()

        if is_tp:
            tp_ac += 1
        else:
            fp_ac += 1

        pre.append(tp_ac / (tp_ac + fp_ac))
        rec.append(tp_ac / gt_num)

    ap_result = 0
    for n in range(len(pre) - 1):
        min_dist_d += pre[n + 1] * (rec[n + 1] - rec[n])
    ap_result = ap_result * 100
    print('AP_RESULT_root: ' + str(ap_result))

