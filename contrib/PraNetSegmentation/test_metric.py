# Copyright (c) 2022. Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
import sys
import json
from argparse import ArgumentParser
import cv2
import numpy as np
import PIL
from PIL import Image
from StreamManagerApi import StreamManagerApi
from tqdm import tqdm
from tabulate import tabulate
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import convolve
import imageio

from main import infer, resize


def object_(pred, gting__):
    xing = np.mean(pred[gting__ == 1])
    sigma_x = np.std(pred[gting__ == 1])
    score = 2.0 * xing / (xing ** 2 + 1 + sigma_x + np.finfo(np.float64).eps)

    return score


def s_object(pred, gting_):
    pred_fg = pred.copy()
    pred_fg[gting_ != 1] = 0.0
    oing_fg = object_(pred_fg, gting_)

    pred_bg = (1 - pred.copy())
    pred_bg[gting_ == 1] = 0.0
    oing_bg = object_(pred_bg, 1-gting_)

    uing = np.mean(gting_)
    qing = uing * oing_fg + (1 - uing) * oing_bg

    return qing


def centroid(gting_1):
    if np.sum(gting_1) == 0:
        return gting_1.shape[0] // 2, gting_1.shape[1] // 2
    xing, ying = np.where(gting_1 == 1)
    return int(np.mean(xing).round()), int(np.mean(ying).round())


def divide(gting_2, xing, ying):
    l_t = gting_2[:xing, :ying]
    r_t = gting_2[xing:, :ying]
    l_b = gting_2[:xing, ying:]
    r_b = gting_2[xing:, ying:]

    wing1 = l_t.size / gting_2.size
    wing2 = r_t.size / gting_2.size
    wing3 = l_b.size / gting_2.size
    wing4 = r_b.size / gting_2.size

    return [l_t, r_t, l_b, r_b, wing1, wing2, wing3, wing4]


def ssim(pred, gting_4):
    xing = np.mean(pred)
    ying = np.mean(gting_4)
    ning_ = pred.size

    sigma_x2 = np.sum((pred - xing) ** 2 /
                      (ning_ - 1 + np.finfo(np.float64).eps))
    sigma_y2 = np.sum((gting_4 - ying) ** 2 /
                      (ning_ - 1 + np.finfo(np.float64).eps))

    sigma_xy = np.sum((pred - xing) * (gting_4 - ying) /
                      (ning_ - 1 + np.finfo(np.float64).eps))

    alpha = 4 * xing * ying * sigma_xy
    beta = (xing ** 2 + ying ** 2) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        qing = alpha / (beta + np.finfo(np.float64).eps)
    elif alpha == 0 and beta == 0:
        qing = 1
    else:
        qing = 0

    return qing


def s_region(pred, gting_5):
    xing, ying = centroid(gting_5)
    gt1, gt2, gt3, gt4, wing1, wing2, wing3, wing4 = divide(
        gting_5, xing, ying)
    pred1, pred2, pred3, pred4, _, _, _, _ = divide(pred, xing, ying)

    qing1 = ssim(pred1, gt1)
    qing2 = ssim(pred2, gt2)
    qing3 = ssim(pred3, gt3)
    qing4 = ssim(pred4, gt4)

    qing = qing1 * wing1 + qing2 * wing2 + qing3 * wing3 + qing4 * wing4

    return qing


def structure_measure(pred, gting_44):
    ying = np.mean(gting_44)

    if ying == 0:
        xing = np.mean(pred)
        qing = 1 - xing
    elif ying == 1:
        xing = np.mean(pred)
        qing = xing
    else:
        alpha = 0.5
        qing = alpha * s_object(pred, gting_44) + \
            (1 - alpha) * s_region(pred, gting_44)
        qing = max(qing, 0)

    return qing


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    xing, ying = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g_1 = np.exp(-((xing**2 + ying**2)/(2.0*sigma**2)))
    return g_1/g_1.sum()


def original_wfb(pred, gting_55):
    eing = np.abs(pred - gting_55)
    dst, idst = distance_transform_edt(1 - gting_55, return_indices=True)

    king = fspecial_gauss(7, 5)
    eting = eing.copy()
    eting[gting_55 != 1] = eting[idst[:, gting_55 != 1]
                                 [0], idst[:, gting_55 != 1][1]]
    eaing = convolve(eting, king, mode='nearest')
    maisfnasdf = eing.copy()
    maisfnasdf[(gting_55 == 1) & (eaing < eing)
               ] = eaing[(gting_55 == 1) & (eaing < eing)]

    bing = np.ones_like(gting_55)
    bing[gting_55 != 1] = 2.0 - 1 * \
        np.exp(np.log(1 - 0.5) / 5 * dst[gting_55 != 1])
    ewing = maisfnasdf * bing

    tpwing = np.sum(gting_55) - np.sum(ewing[gting_55 == 1])
    fpwing = np.sum(ewing[gting_55 != 1])

    ring = 1 - np.mean(ewing[gting_55 == 1])
    ping = tpwing / (tpwing + fpwing + np.finfo(np.float64).eps)
    qing = 2 * ring * ping / (ring + ping + np.finfo(np.float64).eps)

    return qing


def fmeasure_calu(pred, gting_66, threshold_1):
    threshold_1 = min(threshold_1, 1)

    label3 = np.zeros_like(gting_66)
    label3[pred >= threshold_1] = 1

    num_rec = np.sum(label3 == 1)
    num_no_rec = np.sum(label3 == 0)

    label_and = (label3 == 1) & (gting_66 == 1)
    num_and = np.sum(label_and == 1)
    num_obj = np.sum(gting_66)
    num_pred = np.sum(label3)

    fning = num_obj - num_and
    fnmdp = num_rec - num_and
    tning = num_no_rec - fning

    if num_and == 0:
        pre_ftem = 0
        recall_ftem = 0
        f_metric = 0
        dice = 0
        speci_ftem = 0
        iou = 0

    else:
        iou = num_and / (fning + num_rec)
        pre_ftem = num_and / num_rec
        recall_ftem = num_and / num_obj
        speci_ftem = tning / (tning + fnmdp)
        dice = 2 * num_and / (num_obj + num_pred)
        f_metric = ((2.0 * pre_ftem * recall_ftem) / (pre_ftem + recall_ftem))

    return [pre_ftem, recall_ftem, speci_ftem, dice, f_metric, iou]


def alignment_term(pred, gtinga):
    mu_pred = np.mean(pred)
    mu_gt = np.mean(gtinga)

    align_pred = pred - mu_pred
    align_gt = gtinga - mu_gt

    align_mat = 2 * (align_gt * align_pred) / (align_gt ** 2 +
                                               align_pred ** 2 + np.finfo(np.float64).eps)

    return align_mat


def enhanced_alighment_term(align_mat):
    enhanced = ((align_mat + 1) ** 2) / 4
    return enhanced


def enhanced_measure(pred, gtingb):
    if np.sum(gtingb) == 0:
        enhanced_mat = 1 - pred
    elif np.sum(1 - gtingb) == 0:
        enhanced_mat = pred.copy()
    else:
        align_mat = alignment_term(pred, gtingb)
        enhanced_mat = enhanced_alighment_term(align_mat)

    score = np.sum(enhanced_mat) / (gtingb.size - 1 + np.finfo(np.float64).eps)
    return score


def rgb_loader(path):
    with open(path, "rb") as file_22:
        img = Image.open(file_22)
        return img.convert('RGB')


def binary_loader(path):
    with open(path, "rb") as file_33:
        img = Image.open(file_33)
        return img.convert('L')


class TestDataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [
            image_root + f for f in os.listdir(image_root)
            if f.endswith('.jpg') or f.endswith('.png')
        ]
        self.gts = [
            gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.size = len(self.images)
        self.index = 0
        self.mean = np.array(
            [[[0.485]], [[0.456]], [[0.406]]], dtype=np.float32)
        self.std = np.array(
            [[[0.229]], [[0.224]], [[0.225]]], dtype=np.float32)

    def load_data(self):
        image_1 = rgb_loader(self.images[self.index])
        image_1 = resize(image_1, (self.testsize, self.testsize))  # resize
        image_1 = np.transpose(image_1, (2, 0, 1)).astype(
            np.float32)  # to tensor 1
        image_1 = image_1 / 255  # to tensor 2
        image_1 = (image_1 - self.mean) / self.std  # normalize

        gting_222 = binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1

        return [image_1, gting_222, self.images[self.index-1], self.gts[self.index-1]]


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--pipeline_path', type=str,
                        default="pipeline/pranet_pipeline.json")
    parser.add_argument('--data_path', type=str)
    config = parser.parse_args()

    pipeline_path = config.pipeline_path
    data_path = config.data_path

    IMAGESPTH = '{}/images/'.format(data_path)
    GTSPATH = '{}/masks/'.format(data_path)
    dataset = TestDataset(IMAGESPTH, GTSPATH, 352)

    INFER_RESULT = "infer_result/"
    if not os.path.exists(INFER_RESULT):
        os.mkdir(INFER_RESULT)

    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        sys.exit()

    with open(pipeline_path, "r") as file:
        json_str = file.read()

    pipeline = json.loads(json_str)
    pipelineStr = json.dumps(pipeline).encode()

    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        sys.exit()

    Thresholds = np.linspace(1, 0, 256)
    threshold_IoU = np.zeros((dataset.size, len(Thresholds)))
    threshold_Dice = np.zeros((dataset.size, len(Thresholds)))
    Smeasure = np.zeros(dataset.size)
    wFmeasure = np.zeros(dataset.size)
    MAE = np.zeros(dataset.size)

    for i in tqdm(range(dataset.size)):
        image, gting, image_path, gt_path = dataset.load_data()
        gting = np.asarray(gting, np.float32)

        print(image_path)
        image = np.array(image).astype(np.float32)
        infer(image_path, streamManagerApi)

        RESPATH = INFER_RESULT + str(i) + ".png"
        while True:  # 轮询, 等待异步线程
            time.sleep(0.1)
            try:
                pred_mask = np.array(Image.open(RESPATH))
                if not pred_mask.dtype == np.uint8:
                    continue
                break
            except (OSError, FileNotFoundError, PIL.UnidentifiedImageError, SyntaxError):
                pass
        gt_mask = np.array(Image.open(gt_path))

        pred_mask = cv2.resize(pred_mask, dsize=gting.shape)
        pred_mask = pred_mask.transpose(1, 0, 2)

        if len(pred_mask.shape) != 2:
            pred_mask = pred_mask[:, :, 0]
        if len(gt_mask.shape) != 2:
            gt_mask = gt_mask[:, :, 0]

        assert pred_mask.shape == gt_mask.shape

        gt_mask = gt_mask.astype(np.float64) / 255
        gt_mask = (gt_mask > 0.5).astype(np.float64)
        pred_mask = pred_mask.astype(np.float64) / 255

        Smeasure[i] = structure_measure(pred_mask, gt_mask)
        wFmeasure[i] = original_wfb(pred_mask, gt_mask)
        MAE[i] = np.mean(np.abs(gt_mask - pred_mask))

        threshold_E = np.zeros(len(Thresholds))
        threshold_F = np.zeros(len(Thresholds))
        threshold_Pr = np.zeros(len(Thresholds))
        threshold_Rec = np.zeros(len(Thresholds))
        threshold_Iou = np.zeros(len(Thresholds))
        threshold_Spe = np.zeros(len(Thresholds))
        threshold_Dic = np.zeros(len(Thresholds))

        for j, threshold in enumerate(Thresholds):
            threshold_Pr[j], threshold_Rec[j], threshold_Spe[j], \
                threshold_Dic[j], threshold_F[j], threshold_Iou[j] = \
                fmeasure_calu(pred_mask, gt_mask, threshold)

            Bi_pred = np.zeros_like(pred_mask)
            Bi_pred[pred_mask >= threshold] = 1
            threshold_E[j] = enhanced_measure(Bi_pred, gt_mask)

        threshold_Dice[i, :] = threshold_Dic
        threshold_IoU[i, :] = threshold_Iou

    results = []
    result = []

    column_Dic = np.mean(threshold_Dice, axis=0)
    meanDic = np.mean(column_Dic)
    maxDic = np.max(column_Dic)

    column_IoU = np.mean(threshold_IoU, axis=0)
    meanIoU = np.mean(column_IoU)
    maxIoU = np.max(column_IoU)

    result.extend([meanDic, meanIoU])
    results.append(["res", *result])

    headers = ['meanDic', 'meanIoU']
    print(tabulate(results, headers=['dataset', *headers], floatfmt=".3f"))
    print("#"*20, "End Evaluation", "#"*20)

    streamManagerApi.DestroyAllStreams()
