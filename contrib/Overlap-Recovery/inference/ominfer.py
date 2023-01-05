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


import os
import shutil
import warnings
import numpy as np
import cv2
from mindx.sdk import base
from mindx.sdk.base import Tensor, Model, Size, log, ImageProcessor, post, BTensor
from load_img_data import load_img_data
from PIL import Image
warnings.filterwarnings('ignore')

DEVICE_ID = 0  # 芯片ID
MODEL_PATH = "models/best_iou.om"  # 模型的路径
INFER_IMG_PREFIX = './'
IMG_NAME = 'test.jpg'
SAVE_PATH = './'


def om_infer_one(img_name_path, img_prefix=None, vis_dir=None, score_thr=0.4):

    if not os.path.exists(MODEL_PATH):
        print("The input model path is empty!!!")
        print("plz place the model in ./Overlap-Recovery/inference/models/")
        exit()

    base.mx_init() # 全局资源初始化
    model = Model(MODEL_PATH, DEVICE_ID)  # 创造模型对象

    if not os.path.exists(os.path.join(img_prefix, img_name_path)):
        print("The input image path is empty!!!")
        print("plz place the image in ./Overlap-Recovery/inference/")
        exit()

    if cv2.imread(os.path.join(img_prefix, img_name_path)) is None:
        print("=============!Error!================")
        print("The input image is empty, plz check out!")
        print("====================================")
        exit()

    resize_img, img_meta = load_img_data(img_name_path, img_prefix) # hwc-chw
    ori_filename = img_meta['ori_filename']
    abs_filename = img_meta['filename']
    print(f"ori_filename: {img_meta['ori_filename']}")
    print(f"filename: {img_meta['filename']}")
    # h,w,c
    print(f"ori_shape: {img_meta['ori_shape']} "
          f"resize_shape: {img_meta['img_shape']} "
          f"padded_shape: {img_meta['pad_shape']}")
    resize_img = np.expand_dims(resize_img, 0) # add batch dim, 1,3,h,w
    resize_img = np.ascontiguousarray(resize_img)
    image_tensor = Tensor(resize_img) # 推理前需要转换为tensor的List，使用Tensor类来构建。
    image_tensor.to_device(DEVICE_ID) # !!!!!重要，需要转移至device侧，该函数单独执行
    image_tensor_list = [image_tensor] # 推理前需要转换为tensor的List
    outputs = model.infer(image_tensor_list)

    # preds Tensor to numpy
    outputs[0].to_host()
    outputs[0] = np.array(outputs[0])
    outputs[1].to_host()
    outputs[1] = np.array(outputs[1])

    pred_masks,  pred_scores = outputs[0], outputs[1]  # (1, 4, h, w), (1,4) / (1, 4, 1)
    pred_masks,  pred_scores = postprocess(pred_masks, pred_scores)
    print(f"pred_masks_shape: {pred_masks.shape} pred_score_shape: {pred_scores.shape}")
    print(f"original pred unique value: {np.unique(pred_masks)}")

    # remove padding area
    resize_shape = img_meta['img_shape'][:2] # h, w
    pred_masks = pred_masks[:, :, :resize_shape[0], :resize_shape[1]]

    ori_size = img_meta['ori_shape'][:2] # h, w

    # remove batch dim
    pred_masks,  pred_scores = pred_masks[0], pred_scores[0]  # (4, h, w), (4)

    img_id = os.path.basename(ori_filename).split('.')[0]
    if vis_dir is not None:
        save_dir = os.path.join(vis_dir, img_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        shutil.copyfile(abs_filename, os.path.join(save_dir, f"input.{os.path.basename(ori_filename).split('.')[1]}"))
    for instance_idx in range(pred_masks.shape[0]):
        text_instance = pred_masks[instance_idx]
        pred_score = pred_scores[instance_idx]

        if pred_score < score_thr:
            continue

        text_instance = text_instance.astype(np.uint8)
        area = np.sum(text_instance)
        print(f"pred_text_instance: {instance_idx+1} pred_score: {pred_score} "
              f"unique value: {np.unique(text_instance)} area: {area}")

        pred_mask = Image.fromarray(text_instance * 255)
        pred_mask = pred_mask.resize((ori_size[1], ori_size[0]))# w,h

        if vis_dir is not None:
            save_file = os.path.join(save_dir, f'{instance_idx}.png')
            pred_mask.save(save_file, bit=1)
            print(f'pred text mask saving to {save_file}')


def postprocess(scaled_mask_preds, cls_score):
    num_imgs = 1
    segm_results = []
    segm_scores = []
    for img_id in range(num_imgs):
        cls_score_per_img = cls_score[img_id] # num_det, 1
        topk_indices = np.argsort(cls_score_per_img.flatten())[::-1][:4]
        scores_per_img = cls_score_per_img.flatten()[topk_indices]
        mask_indices = topk_indices
        masks_per_img = scaled_mask_preds[img_id][mask_indices] # b, num_det, h,w
        seg_masks = masks_per_img > 0.5
        seg_result, segm_score = segm2result(seg_masks, scores_per_img)
        segm_results.append(seg_result)
        segm_scores.append(segm_score)
    # bs, num_det, h, w
    segm_results = np.stack(segm_results)
    # bs, num_det, 1
    segm_scores = np.stack(segm_scores)
    return segm_results, segm_scores


def segm2result(mask_preds, cls_scores):
    segm_result = []
    seg_scores = []
    num_ins = mask_preds.shape[0] # num_dets, h, w
    for idx in range(num_ins):
        segm_result.append(mask_preds[idx])
        seg_scores.append(cls_scores[idx])
    # here we only have one classes (text)
    segm_result = np.stack(segm_result) # num_det, h, w
    seg_scores = np.stack(seg_scores) # num_det
    return segm_result, seg_scores


if __name__ == '__main__':
    om_infer_one(IMG_NAME, INFER_IMG_PREFIX, vis_dir=SAVE_PATH)

