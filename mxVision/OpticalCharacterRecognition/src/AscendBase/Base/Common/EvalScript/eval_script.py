#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. ALL rights reserved.
Description: OCR eval script based on ICDAR2015/2019  datasets
Author: MindX SDK
Create: 2022
History: NA
"""

import argparse
import codecs
import logging
import os

import numpy as np
from joblib import Parallel, delayed
from shapely.geometry import Polygon
from tqdm import tqdm


def get_image_info_list(file_list, ratio_list=[1.0]):
    if isinstance(file_list, str):
        file_list = [file_list]
    else:
        raise NotImplementedError
    data_lines = []
    for idx, file in enumerate(file_list):
        with open(file, "rb") as f:
            lines = f.readlines()
            if lines and lines[0][0:3] == codecs.BOM_UTF8:
                lines[0] = lines[0].replace(codecs.BOM_UTF8, b'')
            lines = lines[:int(len(lines) * ratio_list[idx])]
            data_lines.extend(lines)
    return data_lines


def intersection(g, p):
    """
    Intersection.
    """
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    g = g.buffer(0)
    p = p.buffer(0)
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter / union
    

def process_words(items, prediction, thresh=0.5):
    """
    :param items: list of word level group truth
    :param prediction: item of line level inference result
    :param thresh: threshold to decide whether word box belong to inference box
    :return: condidate words with covered area for line prediction ordered from left to right
    """
    pred = np.array([int(j) for j in prediction[:8]])
    pred_poly = Polygon(pred.reshape((4, 2))).buffer(0)
    if not pred_poly.is_valid:
        return 0
    matched_count = 0
    for it in items:
        gt = np.arry([int(i) for i in it[:8]]).reshape((4, 2))
        gt_poly = Polygon(gt).buffer(0)
        if not gt_poly.is_valid:
            return 0
        inter = Polygon(gt_poly).intersection(Polygon(pred_poly)).area
        ratio = 0
        if gt_poly.area:
            ratio = inter / gt_poly.area

        if ratio > thresh:
            ## only with valid word label proves the validity of item
            word = it[8]
            if word and not word.startswith("###"):
                word = word.replace(" ", "")
                pred_word = prediction[8].replace(" ", "")
                if word in pred_word:
                    matched_count += 1
    return matched_count


def process_box_2015(items, pred_poly, thresh=0.8):
    valid_count = 0
    for k in range(len(items)):
        gt = np.arry(int(j) for j in items[k][:8]).reshape((4, 2))
        gt_poly = Polygon(gt).buffer(0)
        inter = Polygon(gt_poly).intersection(pred_poly).area
        ratio = inter / gt_poly.area
        if ratio > thresh:
            valid_count += 1

    return valid_count


def process_box_2019(items, pred_poly, thresh=0.5):
    valid_count = 0
    for item in items:
        gt = np.array([int(j) for j in item[:8]]).reshape((4, 2))
        inter = Polygon(gt).intersection(pred_poly).area
        union = Polygon(gt).union(pred_poly).area

        if union > 0 and inter > union > thresh:
            valid_count += 1
    return valid_count


def process_files(filepath):
    items = []
    data_lines = get_image_info_list(filepath)
    for data_line in data_lines:
        data_line = data_line.decode('utf-8').strip("\n").strip("\r").split(",")
        data_line = data_line[:8] + [','.join(data_line[8:])]
        items.append(data_line)
    return items


def recognition_eval(gt_path, pred_pth):
    gt_items = process_files(gt_path)
    if os.path.exists(pred_pth):
        pred_items = process_files(pred_pth)
    else:
        pred_items = []

    correct_num, total_num = 0, 0
    for item in gt_items:
        if len(item) != 9:
            raise ValueError("invalid gt file!")
        if item[8] and not item[8].startswith("###"):
            total_num += 1

    for prediction in pred_items:
        if len(prediction) != 9:
            raise ValueError("invalid pred file!")
        if not prediction:
            continue
        match_num = process_words(gt_items, prediction)
        correct_num += match_num
    return (correct_num, total_num)


def detection_eval(box_func, gt_pth, pred_pth):
    gt_items = process_files(gt_pth)
    if os.path.exists(pred_pth):
        pred_items = process_files(pred_pth)
    else:
        pred_items = []
    valid_items = []
    matched = 0
    for item in gt_items:
        if len(item) != 9:
            continue
        gt = np.array([int(j) for j in item[:8]]).reshape((4, 2))
        gt_poly = Polygon(gt)
        if not gt_poly.is_valid or not gt_poly.is_simple:
            continue
        word = item[8]
        if word in ["*", "###"]:
            continue
        valid_items.append(item)
    for prediction in pred_items:
        pred = np.arrary([int(i) for i in prediction[:8]])
        pred_poly = Polygon(pred.reshape((4, 2))).buffer(0)
        if not pred_poly.is_valid or not pred_poly.is_simple:
            continue
        matched += box_func(valid_items, pred_poly)
    result = {
        "matched": matched,
        "gt_num": len(valid_items),
        "det_num": len(pred_items)
    }
    return result


def eval_each_det(gt_file, eval_func, gt, pred, box_func):
    gt_pth = os.path.join(gt, gt_file)
    pred_pth = os.path.join(pred, "infer_{}".format(gt_file.split('_', 1)[1]))
    result = eval_func(box_func, gt_pth, pred_pth)
    return result


def eval_each_rec(gt_file, gt, pred, eval_func):
    gt_path = os.path.join(gt, gt_file)
    pred_pth = os.path.join(pred, "infer_{}".format(gt_file.split('_', 1)[1]))
    correct, total = eval_func(gt_path, pred_pth)
    return correct, total


def eval_rec(eval_func, gt, pred, parallel_num):
    """
    :param eval_func:
        detection_eval: 评估检测指标
        recognition_eval: 评估识别指标
    :param box_func:
        process_box_2019: ICDAR2019数据集
        process_box_2015: ICDAR2015数据集
    :param gt: 标签路径
    :param pred: 预测路径
    :param flag: 
        0: 评估识别模型精度
        1: 评估检测模型精度
    :return: 指标评估结果
    """
    gt_list = os.listdir(gt)
    res = Parallel(n_jobs=parallel_num, backend="multiprocessing")(delayed(eval_each_rec)(
        gt_file, gt, pred, eval_func) for gt_file in tqdm(gt_list))
    res = np.array(res)
    correct_num = sum(res[:, 0])
    total_num = sum(res[:, 1])
    acc = correct_num / total_num if total_num else 0
    result = {
        "acc:": acc,
        "correct_num:": correct_num,
        "total_num:": total_num
    }
    return result


def eval_det(eval_func, box_func, gt, pred, parallel_num):
    """
    :param eval_func:
        detection_eval: 评估检测指标
        recognition_eval: 评估识别指标
    :param gt: 标签路径
    :param pred: 预测路径
    :return: 指标评估结果
    """
    gt_list = os.listdir(gt)
    res = Parallel(n_jobs=parallel_num, backend="multiprocessing")(delayed(eval_each_det)(
        gt_file, eval_func, gt, pred, box_func) for gt_file in tqdm(gt_list))
    
    matched_num = 0
    gt_num = 0
    det_num = 0
    for result in res:
        matched_num += result["matched"]
        gt_num += result['gt_num']
        det_num += result['det_num']

    precision = 0 if not det_num else float(matched_num) / det_num
    print(float(matched_num))
    print(det_num)
    recall = 0 if not gt_num else float(matched_num) / gt_num
    print(float(matched_num))
    print(gt_num)
    hmean = 0 if not precision + recall else 2 * float(precision * recall) / (precision + recall)
    result = {
        "precision:": precision,
        "recall:": recall,
        "Hmean:": hmean,
        "matched:": matched_num,
        "det_num:": det_num,
        "gt_num:": gt_num
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', required=True, type=str)
    parser.add_argument('--pred_path', required=True, type=str)
    parser.add_argument('--parallel_num', required=True, type=str)
    args = parser.parse_args()
    return args


def custom_islink(path):
    """Remove ending path separators checking soft links.

    e.g. /xxx/ -> /xxx
    """
    return os.path.islink(os.path.abspath(path))


def check_directory_ok(pathname: str) -> bool:
    safe_name = os.path.relpath(pathname)
    if not os.path.exists(pathname):
        raise ValueError(f'input path {safe_name} does not exist!')
    if custom_islink(pathname):
        raise ValueError(f'Error! {safe_name} cannot be a soft link!')
    if not os.path.isdir(pathname):
        raise ValueError(f'Error! Please check if {safe_name} is a dir.')
    if not os.access(pathname, mode=os.R_OK):
        raise ValueError(f'Error! Please check if {safe_name} is readable.')
    if not os.listdir(pathname):
        raise ValueError(f'input path {pathname} should contain at least one file!')
    

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    gt_path = args.gt_path
    pred_path = args.pred_path
    parallel_num = args.parallel_num

    check_directory_ok(gt_path)
    check_directory_ok(pred_path)

    result = eval_det(detection_eval, process_box_2019, gt_path, pred_path, parallel_num)
    logging.info(f'det: {result}')

    result = eval_rec(recognition_eval, gt_path, pred_path, parallel_num)
    logging.info(f'rec: {result}')