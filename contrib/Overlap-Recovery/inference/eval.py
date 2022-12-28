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


import warnings
from PIL import Image
import numpy as np
from mindx.sdk import base
from mindx.sdk.base import Tensor, Model, Size, log, ImageProcessor, post, BTensor
from eval_utils import evaluate_metric
from load_ann import load_annotations
from load_img_data import load_img_data
warnings.filterwarnings('ignore')

DEVICE_ID = 0  # 芯片ID
ANN_FILE_PATH = './dataset/annotation.json'  # 标签路径
IMG_PREFIX_PATH = './dataset'  # 图片根路径
SEG_MASK_PREFIX_PATH = './dataset'  # mask根路径
INFER_MODEL_PATH = "models/best_iou.om"  # 模型的路径


class OverlapDataset:

    def __init__(self, annotation_file, img_prefix_path, seg_prefix):
        self.data_list = load_annotations(annotation_file, img_prefix_path, seg_prefix)
        self.img_prefix = img_prefix_path
        self.seg_prefix = seg_prefix
        self.sample_num = len(self.data_list)
        print(f"There are totally {self.sample_num} samples")

    def __len__(self):
        return self.sample_num

    def __getitem__(self, item):
        data_item = self.data_list[item]
        img_name = data_item['filename']
        img_tensor, img_meta = load_img_data(img_name, self.img_prefix) # hwc-chw
        img_meta['seg_map_path'] = data_item['seg_map_path']
        return img_tensor, img_meta


def prepare_model(model_path, device_id):
    base.mx_init() # 全局资源初始化
    model = Model(model_path, device_id)  # 创造模型对象
    return model


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
    seg_scores =  []
    num_ins = mask_preds.shape[0] # num_dets, h, w
    for idx in range(num_ins):
        segm_result.append(mask_preds[idx])
        seg_scores.append(cls_scores[idx])
    # here we only have one classes (text)
    segm_result = np.stack(segm_result) # num_det, h, w
    seg_scores = np.stack(seg_scores) # num_det
    return segm_result, seg_scores


def evaluate(ann_file, img_prefix, seg_mask_prefix, model_path):
    # dataset
    dataset = OverlapDataset(ann_file, img_prefix, seg_mask_prefix)
    sample_num = dataset.sample_num
    dataset = iter(dataset)

    # model
    model = prepare_model(model_path, DEVICE_ID)

    # inference
    results = []
    img_metas_list = []
    for idx in range(sample_num):
        resize_img, img_meta = next(dataset)
        print(f'sample {idx}')

        # prepare image
        resize_img = np.expand_dims(resize_img, 0)  # add batch dim, 1,3,h,w
        resize_img = np.ascontiguousarray(resize_img)
        image_tensor = Tensor(resize_img)  # 推理前需要转换为tensor的List，使用Tensor类来构建。
        image_tensor.to_device(DEVICE_ID)  # !!!!!重要，需要转移至device侧，该函数单独执行
        image_tensor_list = [image_tensor]  # 推理前需要转换为tensor的List

        # forward
        outputs = model.infer(image_tensor_list)

        # preds Tensor to numpy
        outputs[0].to_host()
        outputs[0] = np.array(outputs[0])
        outputs[1].to_host()
        outputs[1] = np.array(outputs[1])

        pred_masks, pred_scores = outputs[0], outputs[1]  # (1, 4, h, w),  (1, 4, 1)
        pred_masks, pred_scores = postprocess(pred_masks, pred_scores) # (1, 4, h, w), (1, 4)

        # remove padding area
        resize_shape = img_meta['img_shape'][:2]  # h,w
        pred_masks = pred_masks[:, :, :resize_shape[0], :resize_shape[1]]

        # rescaled to original size
        ori_size = img_meta['ori_shape'][:2]  # h,w
        pred_masks = pred_masks[0]  # removed batch dim
        rescaled_masks = []
        for tmp_idx in range(pred_masks.shape[0]):
            img = pred_masks[tmp_idx]
            pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((ori_size[1], ori_size[0]))
            resized_img = np.array(pil_image)
            rescaled_masks.append(resized_img)
        rescaled_masks = np.stack(rescaled_masks)

        rescaled_masks = np.expand_dims(rescaled_masks, 0)
        result = (pred_scores, rescaled_masks)
        results.append(result)
        img_metas_list.append(img_meta)
    # evaluate
    eval_res = evaluate_metric(results, img_metas_list, score_thresh=0.2, )
    text_iou = np.around(eval_res.get("text_iou", 0), decimals=3)
    print("==============================")
    print("精度测试结果如下：")
    print(f'text_iou: {text_iou * 100}%')
    print("==============================")


if __name__ == '__main__':
    evaluate(ANN_FILE_PATH, IMG_PREFIX_PATH, SEG_MASK_PREFIX_PATH, INFER_MODEL_PATH)
