import os
from mmdet.core import bbox2result
from mmdet.core import encode_mask_results
from mmdet.datasets import CocoDataset
import numpy as np

if __name__ == "__main__":
    coco_dataset = CocoDataset(
        ann_file='./dataset/annotations/instances_val2017.json', pipeline=[])
    results = []

    for ids in coco_dataset.img_ids:
        print('image ids = {}'.format(ids))
        bbox_results = []
        # read bbox information
        BBOX_RES_PATH = './binresult/bboxres' + str(ids) + '.bin'
        REBLES_RES_PATH = './binresult/class' + str(ids) + '.bin'
        bboxes = np.fromfile(BBOX_RES_PATH, dtype=np.float32)
        bboxes = np.reshape(bboxes, [100, 5])
        bboxes.tolist()
        # read label information
        labels = np.fromfile(REBLES_RES_PATH, dtype=np.int64)
        labels = np.reshape(labels, [100, 1])
        labels.tolist()
        bbox_results = [bbox2result(bboxes, labels[:, 0], 80)]

        result = bbox_results
        results.extend(result)
    print('Evaluating...')
    eval_results = coco_dataset.evaluate(results,
                                         metric=[
                                             'bbox',
                                         ],
                                         classwise=True)
    print(eval_results)