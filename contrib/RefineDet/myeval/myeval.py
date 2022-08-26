import sys
import os
import time
import numpy as np
import pickle
import cv2
from functools import cmp_to_key


results_objects = []
truth_source = []
truth_images = []
truth_annotations = []
labels_map = ( 
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

aps = []
counts = 0

def voc_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def calc_single_object(d, class_index, current, fp, tp, ovthresh = 0.5):
    global results_objects, truth_annotations

    image = current['image_id']
    bbox = current['bbox']
    confidence = current['score']
    gt_bboxes = [each['bbox'] for each in truth_annotations \
                    if each['category_id'] == class_index and each['image_id'] == image]
    gt = [each for each in truth_annotations \
                    if each['category_id'] == class_index and each['image_id'] == image]

    bbox = np.maximum(bbox, 1.)
    gt_bboxes = np.array(gt_bboxes)
    jmax = 0
    ovmax = -np.inf
    if gt_bboxes.size > 0:
        ixmin = np.maximum(gt_bboxes[:, 0], bbox[0])
        iymin = np.maximum(gt_bboxes[:, 1], bbox[1])
        ixmax = np.minimum(gt_bboxes[:, 2], bbox[2])
        iymax = np.minimum(gt_bboxes[:, 3], bbox[3])
        iw = np.maximum(ixmax - ixmin, 0.)
        ih = np.maximum(iymax - iymin, 0.)
        inters = iw * ih
        uni = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) +
               (gt_bboxes[:, 2] - gt_bboxes[:, 0]) *
               (gt_bboxes[:, 3] - gt_bboxes[:, 1]) - inters)
        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)
        # if ovmax < 0.1:
            # print(image, current['category_id'], gt[jmax]['category_id'], ovmax, confidence)
            # print(bbox, gt_bboxes[jmax])


    if ovmax > 0.5:
        tp[d] = 1. 
    else:
        fp[d] = 1.


def calc_each_class(class_index, current_class):
    global results_objects
    global truth_annotations

    print(current_class)

    class_objects = [each for each in results_objects if each['category_id'] == class_index]
    right_objects = [each for each in truth_annotations if each['category_id'] == class_index]

    num = len(class_objects)


    fp = np.zeros(num)
    tp = np.zeros(num)
    for i, each in enumerate(class_objects):
        calc_single_object(i, class_index, each, fp, tp)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(len(right_objects))
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)

    return ap

def mycmp(A, B):
    return B['score'] - A['score']


def main(): 
    global results_objects
    global truth_source
    global truth_annotations
    global truth_annotations

    filename1 = './precision_analysis/result.json'
    filename2 = './precision_analysis/VOC/VOCdevkit/voc2012val.json'
    
    with open(filename1, "r") as f:
        s = f.read()
        results_objects = eval(s)
    with open(filename2, "r") as f:
        s = f.read()
        truth_source = eval(s)
    truth_images = truth_source['images']
    truth_annotations = truth_source['annotations']

    R = []
    for each in results_objects:
        exist = [e for e in truth_annotations if e['image_id'] == each['image_id']]
        if len(exist) > 0 and each['score'] >= 0.1:
            R.append(each)

    R.sort(key = cmp_to_key(lambda A, B: B['score'] - A['score']))
    results_objects = R

    for i, each_class in enumerate(labels_map):
        ap = calc_each_class(i+1, each_class)
        aps.append(ap)

    print("mAP:", sum(aps)/len(aps))


if __name__ == '__main__':
    main()