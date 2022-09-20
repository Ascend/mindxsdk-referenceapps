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
    ANNOT_PATH_W = 'MuPoTS-3D.json'
    # 测试数据数量
    PIC_NUM = 8400
    with open(OUTPUT_PATH_W, 'r') as f:
        output = json.load(f)

    # AP measure
    
    def return_score(pred):
        return pred['score']


    output.sort(reverse = True, key = return_score)
    dbcoco = COCO(ANNOT_PATH_W)
    G_TAPC = 0
    G_FPAC = 0
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

        G_MINDISTD = 9999
        G_SAVE = -1
        for ann_id, ann in enumerate(anns):
            if ann['is_valid'] == 0:
                continue
            gt_root = np.array(ann['keypoints_cam'])
            G_ROOTDX = 14
            gt_root = gt_root[G_ROOTDX]

            dist_re = math.sqrt(np.sum((pred_inroot - gt_root) ** 2))
            if G_MINDISTD > dist_re:
                G_MINDISTD = dist_re
                G_SAVE = ann_id


        G_ISTP = False
        if G_SAVE != -1 and G_MINDISTD < G_ROOTDX*30:
            try:
                example_value = isare_matched[str(image_idin)][G_SAVE]
                if example_value == 0:
                    G_ISTP = True
                    isare_matched[str(image_idin)][G_SAVE] = 1
            except KeyError:
                print("data error, check ini file!")
                exit()

        if G_ISTP:
            G_TAPC += 1
        else:
            G_FPAC += 1

        pre.append(G_TAPC / (G_TAPC + G_FPAC))
        rec.append(G_TAPC / PIC_NUM)

    G_AP = 0
    for n in range(len(pre) - 1):
        G_AP += pre[n + 1] * (rec[n + 1] - rec[n])
    G_AP = G_AP * 100
    print('AP_RESULT_root: ' + str(G_AP))

