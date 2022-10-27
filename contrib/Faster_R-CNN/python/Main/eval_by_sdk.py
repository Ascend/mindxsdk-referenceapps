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
"""post process for 310 inference"""
import argparse
import json
import os

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

import config as cfg
from util import results2json


def get_img_size(file_name):
    img = Image.open(file_name)
    return img.size


def parse_result(result_file, num_classes):
    all_box = [[] for i in range(0, num_classes)]
    if not os.path.exists(result_file):
        print(f"No such file({result_file}), will be ignore.")
        return [np.asarray(box) for box in all_box]

    with open(result_file, 'r') as fp:
        result = json.loads(fp.read())

    if not result:
        return [np.asarray(box) for box in all_box]

    data = result.get("MxpiObject")
    if not data:
        return [np.asarray(box) for box in all_box]

    for bbox in data:
        class_vec = bbox.get("classVec")[0]
        np_bbox = np.array([
            float(bbox["x0"]),
            float(bbox["y0"]),
            float(bbox["x1"]),
            float(bbox["y1"]),
            class_vec.get("confidence")
        ])
        all_box[int(class_vec["classId"])].append(np_bbox)

    return [np.asarray(box) for box in all_box]


def get_eval_result(ann_file, path):
    outputs = []

    dataset_coco = COCO(ann_file)
    img_ids = dataset_coco.getImgIds()

    for img_id in img_ids:
        image = dataset_coco.loadImgs(img_id)[0]
        file_name = image['file_name'].split('.')[0]
        result_json = os.path.join(path, f"{file_name}.json")
        bbox_results = parse_result(result_json, cfg.NUM_CLASSES)
        outputs.append(bbox_results)

    eval_types = ["bbox"]
    result_files = results2json(dataset_coco, outputs, "./results.pkl")
    result_json_file = "/home/mijianxun1/work/AscendProjects/Faster_Rcnn_App2/sdk2/sdk/results.pkl.bbox.json"


if __name__ == '__main__':
    RESULT_PATH = "./result"
    parser = argparse.ArgumentParser(description="maskrcnn inference")
    parser.add_argument("--ann_file",
                        type=str,
                        required=True,
                        help="ann file.")
    parser.add_argument("--img_path",
                        type=str,
                        required=True,
                        help="image file path.")
    parser.add_argument("--voc_dir",
                        type=str,
                        required=True,
                        help="the path of VOCdevkit")
    args = parser.parse_args()
    get_eval_result(args.ann_file, RESULT_PATH)
