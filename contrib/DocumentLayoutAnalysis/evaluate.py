# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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
import stat
import mindx.sdk as sdk
import numpy as np
from utils import generate_scale, get_result, postprocess, resize_image, decode_image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ANNOTATION_FILE = 'val/annotations.json'
DETECT_FILE = 'val/detection_result.json'
IMAGE_FOLDER = 'val'
MODEL_PATH = 'model/layout.om'


def get_coco_result(images):
    """
    Run this function to get the derivation result of a single image

    Args:
        images: path of detected image

    Returns:
        np_boxes: derivation result of a single image

    """
    device_id = 0
    m = sdk.model(MODEL_PATH, device_id)
    # Preprocess
    im = decode_image(images)
    arr = resize_image(im)
    image = np.expand_dims(arr, axis=0)
    # Gets the image resize ratio
    scale = generate_scale(im)
    t = sdk.Tensor(image)
    t.to_device(0)
    # Make inferences
    outputs = m.infer(t)
    # Postprocess
    out = get_result(outputs)
    the_result = postprocess(scale, out)
    result_boxes = the_result.get('boxes', "abc")
    expect_boxes = (result_boxes[:, 1] > 0.5) & (result_boxes[:, 0] > -1)
    result_boxes = result_boxes[expect_boxes, :]
    return result_boxes


def generate_cocoresult(coco_obj, num):
    """
    Run this function to get the infer output as coco format

    Args:
        coco_obj: the coco object to which the annotation set is converted
        num: the number of image_id

    Returns:
        coco_result: the coco format of the inference result

    """
    coco_result = []
    for image_idx, image_id in enumerate(num):
        image_info = coco_obj.loadImgs(image_id)[0]
        assert os.path.isdir(IMAGE_FOLDER), "This folder does not found"
        image_path = os.path.join(IMAGE_FOLDER, image_info['file_name'])
        print('Detect image: ', image_idx, ': ', image_info['file_name'], ', image id: ', image_id)
        if os.path.exists(image_path) != 1:
            print("The test image does not exist. Exit.")
            exit()
        np_boxes = get_coco_result(image_path)
        for np_box in np_boxes:
            clsid, bbox, score = int(np_box[0]), np_box[2:], np_box[1]
            xmin, ymin, xmax, ymax = bbox
            image_result = {
                'image_id': image_id,
                'category_id': clsid + 1,
                'score': float(score),
                'bbox': [xmin, ymin, xmax - xmin, ymax - ymin]
            }
            coco_result.append(image_result)
    return coco_result


if __name__ == '__main__':
    print('output detection json file path: ', DETECT_FILE)
    coco_gt_obj = COCO(ANNOTATION_FILE)
    image_ids = coco_gt_obj.getImgIds()
    print('Test on val dataset, ', len(image_ids), ' images in total.')
    # get cocoresult
    result = generate_cocoresult(coco_gt_obj, image_ids)
    if os.path.exists(DETECT_FILE):
        os.remove(DETECT_FILE)
    result = json.dumps(result).encode()
    fd = os.open(DETECT_FILE, os.O_RDWR | os.O_CREAT, stat.S_IRWXU | stat.S_IRGRP)
    os.write(fd, result)
    # evaluate process
    coco_dt = coco_gt_obj.loadRes(DETECT_FILE)
    coco_eval = COCOeval(coco_gt_obj, coco_dt, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
