# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://fww.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy.special import softmax


def warp_boxes(boxes):
    width, height = 608, 800
    length = len(boxes)
    box = np.ones((length * 4, 3))
    box[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        length * 4, 2)
    box = (box[:, :2] / box[:, 2:3]).reshape(length, 8)
    x = box[:, [0, 2, 4, 6]]
    y = box[:, [1, 3, 5, 7]]
    box = np.concatenate(
        (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, length).T
    box[:, [0, 2]] = box[:, [0, 2]].clip(0, width)
    box[:, [1, 3]] = box[:, [1, 3]].clip(0, height)
    return box.astype(np.float32)


def generate_nms(box_scores):
    scores = box_scores[:, -1]
    marks = np.argsort(scores)
    marks = marks[-200:]
    length = len(marks)
    all_boxes = box_scores[:, :-1]
    the_pick = []
    while length > 0:
        now_box = all_boxes[marks[-1], :]
        the_pick.append(marks[-1])
        marks = marks[:-1]
        array0 = all_boxes[marks, :]
        array1 = np.expand_dims(now_box, axis=0)
        the_left = np.maximum(array0[..., :2], array1[..., :2])
        the_right = np.minimum(array0[..., 2:], array1[..., 2:])
        clip1 = np.clip(the_right - the_left, 0.0, None)
        clip2 = np.clip(array0[..., 2:] - array0[..., :2], 0.0, None)
        clip3 = np.clip(array1[..., 2:] - array1[..., :2], 0.0, None)
        the_area = clip1[..., 0] * clip1[..., 1]
        area0 = clip2[..., 0] * clip2[..., 1]
        area1 = clip3[..., 0] * clip3[..., 1]
        mask = the_area / (area0 + area1 - the_area + 1e-5)
        marks = marks[mask <= 0.5]
        length = len(marks)
    return box_scores[the_pick, :]


def generate_centers(the_id, result_boxes, scores):
    input_shape = np.array([800, 608]).astype('float32')
    reg_max = int(result_boxes[0].shape[-1] / 4 - 1)
    strides = [8, 16, 32, 64]
    all_boxes = []
    the_grade = []
    for stride, divide_box, grade in zip(strides, result_boxes,
                                         scores):
        divide_box = divide_box[the_id]
        space_box = divide_box.reshape((-1, reg_max + 1))
        space_box = softmax(space_box, axis=1)
        reg_range = np.arange(reg_max + 1)
        space_box = space_box * np.expand_dims(reg_range, axis=0)
        space_box = np.sum(space_box, axis=1).reshape((-1, 4))
        space_box = space_box * stride
        grade = grade[the_id]
        heigth = input_shape[0] / stride
        width = input_shape[1] / stride
        height_range = np.arange(heigth)
        width_range = np.arange(width)
        fw, fh = np.meshgrid(width_range, height_range)
        ct_row = (fh.flatten() + 0.5) * stride
        ct_col = (fw.flatten() + 0.5) * stride
        the_center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)
        top_flag = np.argsort(grade.max(axis=1))[::-1]
        top_flag = top_flag[:1000]
        the_center = the_center[top_flag]
        grade = grade[top_flag]
        space_box = space_box[top_flag]
        decode = the_center + [-1, -1, 1, 1] * space_box
        the_grade.append(grade)
        all_boxes.append(decode)
    return all_boxes, the_grade


def generate_output(scale, indexs, decode, choose):
    output_id = []
    output_boxes = []
    get_box = []
    labels = []
    the_boxes = np.concatenate(decode, axis=0)
    verify = np.concatenate(choose, axis=0)
    for index in range(0, verify.shape[1]):
        probs = verify[:, index]
        mask = probs > 0.4
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = the_boxes[mask, :]
        box_probs = np.concatenate(
            [subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = generate_nms(box_probs)
        get_box.append(box_probs)
        labels.extend([index] * box_probs.shape[0])
    if len(get_box) == 0:
        output_boxes.append(np.empty((0, 4)))
        output_id.append(0)
    else:
        get_box = np.concatenate(get_box)
        get_box[:, :4] = warp_boxes(get_box[:, :4])
        scale1 = scale[indexs][::-1]
        sacle2 = scale[indexs][::-1]
        im_scale = np.concatenate([scale1, sacle2])
        get_box[:, :4] /= im_scale
        dim1 = np.expand_dims(np.array(labels), axis=-1)
        dim2 = np.expand_dims(get_box[:, 4], axis=-1)
        other = get_box[:, :4]
        output_boxes.append(np.concatenate([dim1, dim2, other], axis=1))
        output_id.append(len(labels))
    return output_boxes, output_id
