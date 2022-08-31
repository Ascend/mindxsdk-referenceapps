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
import json
import os
from argparse import ArgumentParser
import cv2
import numpy as np
from PIL import Image
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector, InProtobufVector, MxProtobufIn
from tqdm import tqdm
from tabulate import tabulate
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import convolve
import imageio

from infer import infer, resize

def Object(pred, gt):
    x = np.mean(pred[gt == 1])
    sigma_x = np.std(pred[gt == 1])
    score = 2.0 * x / (x ** 2 + 1 + sigma_x + np.finfo(np.float64).eps)

    return score

def S_Object(pred, gt):
    pred_fg = pred.copy()
    pred_fg[gt != 1] = 0.0
    O_fg = Object(pred_fg, gt)
    
    pred_bg = (1 - pred.copy())
    pred_bg[gt == 1] = 0.0
    O_bg = Object(pred_bg, 1-gt)

    u = np.mean(gt)
    Q = u * O_fg + (1 - u) * O_bg

    return Q

def centroid(gt):
    if np.sum(gt) == 0:
        return gt.shape[0] // 2, gt.shape[1] // 2
    
    else:
        x, y = np.where(gt == 1)
        return int(np.mean(x).round()), int(np.mean(y).round())

def divide(gt, x, y):
    LT = gt[:x, :y]
    RT = gt[x:, :y]
    LB = gt[:x, y:]
    RB = gt[x:, y:]

    w1 = LT.size / gt.size
    w2 = RT.size / gt.size
    w3 = LB.size / gt.size
    w4 = RB.size / gt.size

    return LT, RT, LB, RB, w1, w2, w3, w4

def ssim(pred, gt):
    x = np.mean(pred)
    y = np.mean(gt)
    N = pred.size

    sigma_x2 = np.sum((pred - x) ** 2 / (N - 1 + np.finfo(np.float64).eps))
    sigma_y2 = np.sum((gt - y) ** 2 / (N - 1 + np.finfo(np.float64).eps))

    sigma_xy = np.sum((pred - x) * (gt - y) / (N - 1 + np.finfo(np.float64).eps))

    alpha = 4 * x * y * sigma_xy
    beta = (x ** 2 + y ** 2) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        Q = alpha / (beta + np.finfo(np.float64).eps)
    elif alpha == 0 and beta == 0:
        Q = 1
    else:
        Q = 0
    
    return Q

def S_Region(pred, gt):
    x, y = centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = divide(gt, x, y)
    pred1, pred2, pred3, pred4, _, _, _, _ = divide(pred, x, y)

    Q1 = ssim(pred1, gt1)
    Q2 = ssim(pred2, gt2)
    Q3 = ssim(pred3, gt3)
    Q4 = ssim(pred4, gt4)

    Q = Q1 * w1 + Q2 * w2 + Q3 * w3 + Q4 * w4

    return Q

def StructureMeasure(pred, gt):
    y = np.mean(gt)

    if y == 0:
        x = np.mean(pred)
        Q = 1 - x
    elif y == 1:
        x = np.mean(pred)
        Q = x
    else:
        alpha = 0.5
        Q = alpha * S_Object(pred, gt) + (1 - alpha) * S_Region(pred, gt)
        if Q < 0:
            Q = 0
    
    return Q

def fspecial_gauss(size, sigma):
       """Function to mimic the 'fspecial' gaussian MATLAB function
       """
       x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
       g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
       return g/g.sum()

def original_WFb(pred, gt):
    E = np.abs(pred - gt)
    dst, idst = distance_transform_edt(1 - gt, return_indices=True)

    K = fspecial_gauss(7, 5)
    Et = E.copy()
    Et[gt != 1] = Et[idst[:, gt != 1][0], idst[:, gt != 1][1]]
    EA = convolve(Et, K, mode='nearest')
    MIN_E_EA = E.copy()
    MIN_E_EA[(gt == 1) & (EA < E)] = EA[(gt == 1) & (EA < E)]

    B = np.ones_like(gt)
    B[gt != 1] = 2.0 - 1 * np.exp(np.log(1 - 0.5) / 5 * dst[gt != 1])
    Ew = MIN_E_EA * B

    TPw = np.sum(gt) - np.sum(Ew[gt == 1])
    FPw = np.sum(Ew[gt != 1])

    R = 1 - np.mean(Ew[gt == 1])
    P = TPw / (TPw + FPw + np.finfo(np.float64).eps)
    Q = 2 * R * P / (R + P + np.finfo(np.float64).eps)

    return Q

def Fmeasure_calu(pred, gt, threshold):
    if threshold > 1:
        threshold = 1

    Label3 = np.zeros_like(gt)
    Label3[pred >= threshold] = 1

    NumRec = np.sum(Label3 == 1)
    NumNoRec = np.sum(Label3 == 0)

    LabelAnd = (Label3 == 1) & (gt == 1)
    NumAnd = np.sum(LabelAnd == 1)
    num_obj = np.sum(gt)
    num_pred = np.sum(Label3)

    FN = num_obj - NumAnd
    FP = NumRec - NumAnd
    TN = NumNoRec - FN

    if NumAnd == 0:
        PreFtem = 0
        RecallFtem = 0
        FmeasureF = 0
        Dice = 0
        SpecifTem = 0
        IoU = 0

    else:
        IoU = NumAnd / (FN + NumRec)
        PreFtem = NumAnd / NumRec
        RecallFtem = NumAnd / num_obj
        SpecifTem = TN / (TN + FP)
        Dice = 2 * NumAnd / (num_obj + num_pred)
        FmeasureF = ((2.0 * PreFtem * RecallFtem) / (PreFtem + RecallFtem))
    
    return PreFtem, RecallFtem, SpecifTem, Dice, FmeasureF, IoU

def AlignmentTerm(pred, gt):
    mu_pred = np.mean(pred)
    mu_gt = np.mean(gt)

    align_pred = pred - mu_pred
    align_gt = gt - mu_gt

    align_mat = 2 * (align_gt * align_pred) / (align_gt ** 2 + align_pred ** 2 + np.finfo(np.float64).eps)
    
    return align_mat

def EnhancedAlighmentTerm(align_mat):
    enhanced = ((align_mat + 1) ** 2) / 4
    return enhanced

def EnhancedMeasure(pred, gt):
    if np.sum(gt) == 0:
        enhanced_mat = 1 - pred
    elif np.sum(1 - gt) == 0:
        enhanced_mat = pred.copy()
    else:
        align_mat = AlignmentTerm(pred, gt)
        enhanced_mat = EnhancedAlighmentTerm(align_mat)
    
    score = np.sum(enhanced_mat) / (gt.size - 1 + np.finfo(np.float64).eps)
    return score

class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.size = len(self.images)
        self.index = 0
        self.mean = np.array([[[0.485]], [[0.456]], [[0.406]]], dtype=np.float32)
        self.std = np.array([[[0.229]], [[0.224]], [[0.225]]], dtype=np.float32)

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = resize(image, (self.testsize, self.testsize)) # resize
        image = np.transpose(image, (2,0,1)).astype(np.float32) # to tensor 1
        image = image / 255 # to tensor 2
        image = (image - self.mean) / self.std # normalize

        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1

        return image, gt, self.images[self.index-1], self.gts[self.index-1]

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--pipeline_path', type=str)
    parser.add_argument('--data_path', type=str)
    config = parser.parse_args()

    result_path = "./"
    pipeline_path = config.pipeline_path
    data_path = config.data_path

    images_path = '{}/images/'.format(data_path)
    gts_path = '{}/masks/'.format(data_path)
    dataset = test_dataset(images_path, gts_path, 352)

    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    with open(pipeline_path, "r") as file:
        json_str = file.read()
    pipeline = json.loads(json_str)
    pipelineStr = json.dumps(pipeline).encode()

    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    Thresholds = np.linspace(1, 0, 256)
    threshold_IoU = np.zeros((dataset.size, len(Thresholds)))
    threshold_Dice = np.zeros((dataset.size, len(Thresholds)))
    Smeasure = np.zeros(dataset.size)
    wFmeasure = np.zeros(dataset.size)
    MAE = np.zeros(dataset.size)

    for i in tqdm(range(dataset.size)):
        image, gt, image_path, gt_path = dataset.load_data()
        gt = np.asarray(gt, np.float32)

        image = np.array(image).astype(np.float32)
        res = infer(image.tobytes(), streamManagerApi)
        res = np.reshape(res, (1, 1, 352, 352))
        res = res.reshape((352, 352))
        res = cv2.resize(res.T, dsize=gt.shape)
        res = res.T
        res = np.expand_dims(res, 0)
        res = np.expand_dims(res, 0)
        res = 1 / (1 + np.exp(-res))
        res = res.squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        imageio.imwrite("temp.png", res)
        pred_mask = np.array(Image.open("temp.png"))
        gt_mask = np.array(Image.open(gt_path))

        if len(pred_mask.shape) != 2:
            pred_mask = pred_mask[:, :, 0]
        if len(gt_mask.shape) != 2:
            gt_mask = gt_mask[:, :, 0]
        assert pred_mask.shape == gt_mask.shape

        gt_mask = gt_mask.astype(np.float64) / 255
        gt_mask = (gt_mask > 0.5).astype(np.float64)
        pred_mask = pred_mask.astype(np.float64) / 255

        Smeasure[i] = StructureMeasure(pred_mask, gt_mask)
        wFmeasure[i] = original_WFb(pred_mask, gt_mask)
        MAE[i] = np.mean(np.abs(gt_mask - pred_mask))

        threshold_E = np.zeros(len(Thresholds))
        threshold_F = np.zeros(len(Thresholds))
        threshold_Pr = np.zeros(len(Thresholds))
        threshold_Rec = np.zeros(len(Thresholds))
        threshold_Iou = np.zeros(len(Thresholds))
        threshold_Spe = np.zeros(len(Thresholds))
        threshold_Dic = np.zeros(len(Thresholds))

        for j, threshold in enumerate(Thresholds):
            threshold_Pr[j], threshold_Rec[j], threshold_Spe[j], threshold_Dic[j], threshold_F[j], threshold_Iou[j] = Fmeasure_calu(pred_mask, gt_mask, threshold)

            Bi_pred = np.zeros_like(pred_mask)
            Bi_pred[pred_mask >= threshold] = 1
            threshold_E[j] = EnhancedMeasure(Bi_pred, gt_mask)

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
    
    json = os.path.join(result_path,'result_'+"res"+'.json')
    json = open(json, 'w')

    headers = ['meanDic', 'meanIoU']
    csv = os.path.join(result_path, 'result_' + "res" + '.csv')
    if os.path.isfile(csv) is True:
        csv = open(csv, 'a')
    else:
        csv = open(csv, 'w')
        csv.write(', '.join(['method', *headers]) + '\n')

    method = "pranet"
    out_str = method + ','
    for metric in result:
        out_str += '{:.4f}'.format(metric) + ','
    out_str += '\n'

    csv.write(out_str)
    csv.close()
    json.write(out_str)
    json.close()

    print(tabulate(results, headers=['dataset', *headers], floatfmt=".3f"))
    print("#"*20, "End Evaluation", "#"*20)

    streamManagerApi.DestroyAllStreams()
