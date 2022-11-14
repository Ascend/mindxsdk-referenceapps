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

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from postprocess import generate_centers, generate_output


def decode_image(image):
    """
    read rgb image

    Args:
        im_file (str|np.ndarray): input can be image path or np.ndarray

    Returns:
        im (np.ndarray):  processed image (np.ndarray)

    """
    with open(image, 'rb') as f:
        im_read = f.read()
        data = np.frombuffer(im_read, dtype='uint8')
        try:
            im = cv2.imdecode(data, 1)
        except(cv2.error):
            print("error: The input image cannot be decode, may not be the correct image, the program stops")
            exit()
    return im


def resize_image(image):
    """
    resize image by target_size

    Args:
        target_size (int): the target size of image

    Returns:
        im :  processed image

    """
    origin_shape = image.shape[:2]
    resize_h, resize_w = 800, 608
    im_scale_y = resize_h / float(origin_shape[0])
    im_scale_x = resize_w / float(origin_shape[1])
    im = cv2.resize(
        image,
        None,
        None,
        fx=im_scale_x,
        fy=im_scale_y,
        interpolation=2)
    return im


def get_result(outputs):
    """
    run this function to Process inference results

    Args:
        outputs: inference results

    Returns:
        result: includes boxes and boxes_num composed of dict types

    """
    np_score_list, np_boxes_list = [], []
    for i in range(0, 4):
        outputs[i].to_host()
        array = np.array(outputs[i])
        np_score_list.append(array)
    for i in range(4, 8):
        outputs[i].to_host()
        array = np.array(outputs[i])
        np_boxes_list.append(array)
    result = dict(boxes=np_score_list, boxes_num=np_boxes_list)
    return result


def generate_scale(im):
    """
    run this function to get the picture zoom factor

    Args:
        im: the decoded picture

    Returns:
        [scale_factor]: a ratio data of list

    """
    origin_shape = im.shape[:2]
    resize_h, resize_w = 800, 608
    im_scale_y = resize_h / float(origin_shape[0])
    im_scale_x = resize_w / float(origin_shape[1])
    scale_factor = np.array([im_scale_y, im_scale_x]).astype('float32')
    return [scale_factor]


def postprocess(scale, result):
    """
    run this function to post-processing inference results

    Args:
        result: inference results
        scale: image zoom ratio

    Returns:
        result: includes boxes and boxes_num composed of dict types

    """
    np_score_list = result['boxes']
    np_boxes_list = result['boxes_num']
    the_size = np_boxes_list[0].shape[0]
    for index in range(the_size):
        decode_boxes, choose_scores = generate_centers(index, np_boxes_list, np_score_list)
        output0, output1 = generate_output(scale, index, decode_boxes, choose_scores)
    output0 = np.concatenate(output0, axis=0)
    output1 = np.asarray(output1).astype(np.int32)
    result = dict(boxes=output0, boxes_num=output1)
    return result


def visualize(image, result, labels, output_dir):
    """
    run this function to visualize post-processing results

    Args:
        image: the input image
        result: post-processing results
        labels: text recognition label files
        output_dir: where the visualization results are saved

    Returns:
        None

    """
    np_boxes = result['boxes']
    im = image
    threshold = 0.5
    clsid2color = {}
    im = Image.open(im).convert('RGB')
    draw_thickness = min(im.size) // 320
    draw = ImageDraw.Draw(im)
    color_map = 10 * [0, 0, 0]
    for i in range(0, 10):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_list = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
    np_boxes = np_boxes[expect_boxes, :]
    for dt in np_boxes:
        clsid, bbox, score = int(dt[0]), dt[2:], dt[1]
        if clsid not in clsid2color:
            clsid2color[clsid] = color_list[clsid]
        color = tuple(clsid2color.get(clsid, "abc"))
        xmin, ymin, xmax, ymax = bbox
        print("analyze results:", labels[int(clsid)], end=", ")
        print('confidence:{:.4f}, left_top:[{:.2f},{:.2f}], '
              'right_bottom:[{:.2f},{:.2f}]'.format(
            score, xmin, ymin, xmax, ymax))
        # draw bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=draw_thickness,
            fill=color)
        # draw label
        text = "{} {:.4f}".format(labels[clsid], score)
        tw, th = draw.textsize(text)
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))
    img_name = os.path.split(image)[-1]
    out_path = os.path.join(output_dir, img_name)
    im.save(out_path, quality=95)
    print("save result to: " + out_path + "\n")
