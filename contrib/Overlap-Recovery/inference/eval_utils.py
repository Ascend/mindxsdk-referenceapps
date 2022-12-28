#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import cv2


def cal_mask_iou(mask_a, mask_b, check_valid=False):
    if check_valid:
        assert len(np.unique(mask_a)) <= 2
        assert len(np.unique(mask_b)) <= 2
    a_bool = mask_a.astype(np.bool)
    b_bool = mask_b.astype(np.bool)
    intersection_area = (a_bool & b_bool).sum()
    union_area = (a_bool | b_bool).sum()
    if union_area == 0:
        return 0
    return intersection_area / union_area


def cal_overlap_mask(mask_list):
    if len(mask_list) < 2:
        return None
    mask_list_bool = [x.astype(np.bool) for x in mask_list]
    overlap_mask = np.zeros_like(mask_list_bool[0])
    for ii in range(len(mask_list_bool) - 1):
        for jj in range(ii + 1, len(mask_list_bool)):
            cur_olp = mask_list_bool[ii] & mask_list_bool[jj]
            overlap_mask = overlap_mask | cur_olp
    return overlap_mask


def cal_union_mask(mask_list):
    if len(mask_list) < 1:
        return None
    mask_list_bool = [x.astype(np.bool) for x in mask_list]
    union_mask = np.zeros_like(mask_list_bool[0])
    for mask_bool in mask_list_bool:
        union_mask = union_mask | mask_bool
    return union_mask



def eval_func(box_scores, masks, img_meta, score_thresh=0.2, iou_thresh=0.5):
    # prepare gt
    gt_masks = [cv2.imread(x, cv2.IMREAD_UNCHANGED) // 255 for x in img_meta['seg_map_path']]
    for mask_ in gt_masks:
        if len(mask_.shape) > 2:
            import ipdb
            ipdb.set_trace()
            print(gt_masks)
    gt_text = cal_union_mask(gt_masks)
    gt_overlap = cal_overlap_mask(gt_masks)
    # prepare predict of overlap and text area

    # select top 2 prediction
    box_scores = box_scores[0] # remove batch dim
    scores = box_scores.tolist()
    valid_idx = []
    for ins_idx, score in enumerate(box_scores):
        if score > score_thresh:
            valid_idx.append(ins_idx)
    pred_masks = [masks[0][_] for _ in valid_idx]
    if len(pred_masks) == 0:
        pred_overlap = np.zeros_like(masks[0][0])
        pred_text = np.zeros_like(masks[0][0])
    elif len(pred_masks) == 1:
        pred_overlap = np.zeros_like(masks[0][0])
        pred_text = cal_union_mask(pred_masks)
    else:
        pred_overlap = cal_overlap_mask(pred_masks)
        pred_text = cal_union_mask(pred_masks)

    if len(gt_masks) > 1:
        # calculate metrics
        intersection_text = (pred_text & gt_text).sum()
        union_text = (pred_text | gt_text).sum()
        intersection_overlap = (pred_overlap & gt_overlap).sum()
        union_overlap = (pred_overlap | gt_overlap).sum()
    else:
        intersection_text = 0
        union_text = 0
        intersection_overlap = 0
        union_overlap = 0

    # prepare predict of text instance
    # filter out invalid prediction
    valid_idx = []
    for ins_idx, score in enumerate(box_scores):
        if score > score_thresh:
            valid_idx.append(ins_idx)
    match_matrix = np.zeros((len(valid_idx), len(gt_masks)), dtype=np.bool)
    for ins_idx, tmp_valid_idx in enumerate(valid_idx):
        for gt_ins_idx, tmp_gt_mask in enumerate(gt_masks):
            if match_matrix[:, gt_ins_idx].sum() > 0:
                continue
            # calculate IoU
            if cal_mask_iou(masks[0][tmp_valid_idx], tmp_gt_mask) > iou_thresh:
                match_matrix[ins_idx, gt_ins_idx] = True
                break
    # calculate instance-wise mIoU
    text_ins_miou = 0
    if match_matrix.sum() > 0:
        for ins_idx in range(max(match_matrix.shape)):
            if ins_idx >= match_matrix.shape[0]:
                # miss det
                continue
            else:
                if ins_idx >= match_matrix.shape[1] or match_matrix[ins_idx].sum() == 0:
                    # wrong det
                    continue
                else:
                    pred_mask = masks[0][valid_idx[ins_idx]].astype(np.bool)
                    gt_idx = match_matrix[ins_idx].nonzero()[0][0]
                    gt_mask = gt_masks[gt_idx].copy()
                    cur_iou = cal_mask_iou(pred_mask, gt_mask)
                    text_ins_miou += cur_iou
    return (intersection_text, union_text, intersection_overlap, union_overlap), \
           text_ins_miou, max(match_matrix.shape)


def evaluate_metric(results,
                    img_metas,
                    score_thresh=0.2,
                    iou_thrs=0.5,
                    ):

    intersection_text = 0
    union_text = 0
    intersection_overlap = 0
    union_overlap = 0
    text_ins_miou_list = []
    total_ins_num = 0
    for idx, ((box_scores, masks), img_meta) in enumerate(zip(results, img_metas)):
        overall_iou_metrics, text_ins_miou, ins_num = eval_func(box_scores, masks, img_meta, score_thresh, iou_thrs)
        intersection_text += overall_iou_metrics[0]
        union_text += overall_iou_metrics[1]
        intersection_overlap += overall_iou_metrics[2]
        union_overlap += overall_iou_metrics[3]
        text_ins_miou_list.append(text_ins_miou)
        total_ins_num += ins_num

    metric_results = dict(
        text_iou=intersection_text / union_text,
    )

    return metric_results

