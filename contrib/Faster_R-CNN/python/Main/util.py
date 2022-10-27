# Copyright 2022 Huawei Technologies Co., Ltd
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
# ============================================================================
"""coco eval for maskrcnn"""
import os
import json
import shutil
import xml.etree.ElementTree as ET
import mmcv
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import cv2 as cv
import tqdm
import matplotlib.pyplot as plt
from pycocotools.coco import COCO


_init_value = np.array(0.0)
summary_init = {
    'Precision/mAP': _init_value,
    'Precision/mAP@.50IOU': _init_value,
    'Precision/mAP@.75IOU': _init_value,
    'Precision/mAP (small)': _init_value,
    'Precision/mAP (medium)': _init_value,
    'Precision/mAP (large)': _init_value,
    'Recall/AR@1': _init_value,
    'Recall/AR@10': _init_value,
    'Recall/AR@100': _init_value,
    'Recall/AR@100 (small)': _init_value,
    'Recall/AR@100 (medium)': _init_value,
    'Recall/AR@100 (large)': _init_value,
}


def coco_eval(result_files,
              result_types,
              coco,
              max_dets=(100, 300, 1000),
              single_result=False):
    """coco eval for maskrcnn"""
    anns = json.load(open(result_files['bbox']))
    if not anns:
        return summary_init
    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    for res_type in result_types:
        result_file = result_files[res_type]
        assert result_file.endswith('.json')

        coco_dets = coco.loadRes(result_file)
        gt_img_ids = coco.getImgIds()
        det_img_ids = coco_dets.getImgIds()
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        coco_eval = COCOeval(coco, coco_dets, iou_type)
        if res_type == 'proposal':
            coco_eval.params.useCats = 0
            coco_eval.params.maxDets = list(max_dets)

        tgt_ids = gt_img_ids if not single_result else det_img_ids

        if single_result:
            res_dict = dict()
            for id_i in tgt_ids:
                coco_eval = COCOeval(coco, coco_dets, iou_type)
                if res_type == 'proposal':
                    coco_eval.params.useCats = 0
                    coco_eval.params.maxDets = list(max_dets)

                coco_eval.params.imgIds = [id_i]
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                res_dict.update(
                    {coco.imgs[id_i]['file_name']: coco_eval.stats[1]})

        coco_eval = COCOeval(coco, coco_dets, iou_type)
        if res_type == 'proposal':
            coco_eval.params.useCats = 0
            coco_eval.params.maxDets = list(max_dets)

        coco_eval.params.imgIds = tgt_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        summary_metrics = {
            'Precision/mAP': coco_eval.stats[0],
            'Precision/mAP@.50IOU': coco_eval.stats[1],
            'Precision/mAP@.75IOU': coco_eval.stats[2],
            'Precision/mAP (small)': coco_eval.stats[3],
            'Precision/mAP (medium)': coco_eval.stats[4],
            'Precision/mAP (large)': coco_eval.stats[5],
            'Recall/AR@1': coco_eval.stats[6],
            'Recall/AR@10': coco_eval.stats[7],
            'Recall/AR@100': coco_eval.stats[8],
            'Recall/AR@100 (small)': coco_eval.stats[9],
            'Recall/AR@100 (medium)': coco_eval.stats[10],
            'Recall/AR@100 (large)': coco_eval.stats[11],
        }

    print(json.dumps(summary_metrics, indent=2))

    return summary_metrics


def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]


def det2json(dataset, results):
    """convert det to json"""
    cat_ids = dataset.getCatIds()
    img_ids = dataset.getImgIds()
    json_results = []
    dataset_len = len(img_ids)
    for idx in range(dataset_len):
        img_id = img_ids[idx]
        if idx == len(results):
            break
        result = results[idx]
        for label, result_label in enumerate(result):
            bboxes = result_label
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = cat_ids[label]
                json_results.append(data)
    return json_results


def results2json(dataset, results, out_file):
    """convert result to json"""
    result_files = dict()
    json_results = det2json(dataset, results)
    result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
    result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
    mmcv.dump(json_results, result_files['bbox'])
    return result_files
