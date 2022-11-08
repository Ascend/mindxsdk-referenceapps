#!/usr/bin/env python
# coding=utf-8

"""
Copyright(C) Huawei Technologies Co.,Ltd. 2022 All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from ast import mod
import os
import time
import numpy as np
from tqdm import tqdm
from numpy.lib.stride_tricks import as_strided
import yaml
from PIL import Image

import cv2


import mindx.sdk as sdk

CONFIG_PATH = './config/test.yaml' # config path
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
else:
    raise FileNotFoundError("Config File doesn't Exist")

model_image_width = config['PREDICT']['model_image_width']
model_image_height = config['PREDICT']['model_image_height']
nms_thresh = config['PREDICT']['nms_thresh']
nms_size = config['PREDICT']['nms_size']
knn_thresh = config['PREDICT']['knn_thresh']
use_matching_trick = config['PREDICT']['use_matching_trick']


def pre_processing(data):
    """ Enhance retinal images """
    train_imgs = datasets_normalized(data)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)

    train_imgs = train_imgs / 255.

    return train_imgs.astype(np.float32)


def datasets_normalized(images):
    images_std = np.std(images)
    images_mean = np.mean(images)
    images_normalized = (images - images_mean) / (images_std + 1e-6)
    minv = np.min(images_normalized)
    images_normalized = ((images_normalized - minv) /
                         (np.max(images_normalized) - minv)) * 255

    return images_normalized


def clahe_equalized(images):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    images_equalized = np.empty(images.shape)
    images_equalized[:, :] = clahe.apply(np.array(images[:, :],
                                                  dtype=np.uint8))

    return images_equalized


def adjust_gamma(images, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    new_images = np.empty(images.shape)
    new_images[:, :] = cv2.LUT(np.array(images[:, :],
                                        dtype=np.uint8), table)

    return new_images


def transform(image):
    img = Image.fromarray(image)
    img = img.resize((768, 768), resample=Image.BILINEAR) # input image size must be 768x768
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float64) / 255.0
    return img


def image_read(read_query_path, read_refer_path, query_is_image=False):
    if query_is_image:
        read_query_image = read_query_path
    else:
        read_query_image = cv2.imread(read_query_path, cv2.IMREAD_COLOR)
        read_query_image = read_query_image[:, :, 1]
        read_query_image = pre_processing(read_query_image)
    read_refer_image = cv2.imread(read_refer_path, cv2.IMREAD_COLOR)

    assert read_query_image.shape[:2] == read_refer_image.shape[:2]
    read_image_height, read_image_width = read_query_image.shape[:2]

    read_refer_image = read_refer_image[:, :, 1]
    read_refer_image = pre_processing(read_refer_image)

    read_query_image = (read_query_image * 255).astype(np.uint8)
    read_refer_image = (read_refer_image * 255).astype(np.uint8)

    return [read_query_image, read_refer_image, read_image_height, read_image_width]


def cal_auc(s_error, p_error, a_error):
    assert (len(s_error) == 71)  # Easy pairs
    assert (len(p_error) == 48)  # Hard pairs. Note file control_points_P37_1_2.txt is ignored
    assert (len(a_error) == 14)  # Moderate pairs

    s_error = np.array(s_error)
    p_error = np.array(p_error)
    a_error = np.array(a_error)

    limit = 25
    gs_error = np.zeros(limit + 1)
    gp_error = np.zeros(limit + 1)
    ga_error = np.zeros(limit + 1)

    accum_s = 0
    accum_p = 0
    accum_a = 0

    for i in range(1, limit + 1):
        gs_error[i] = np.sum(s_error < i) * 100 / len(s_error)
        gp_error[i] = np.sum(p_error < i) * 100 / len(p_error)
        ga_error[i] = np.sum(a_error < i) * 100 / len(a_error)

        accum_s = accum_s + gs_error[i]
        accum_p = accum_p + gp_error[i]
        accum_a = accum_a + ga_error[i]

    auc_s = accum_s / (limit * 100)
    auc_p = accum_p / (limit * 100)
    auc_a = accum_a / (limit * 100)
    mauc = (auc_s + auc_p + auc_a) / 3.0
    return {'s': auc_s, 'p': auc_p, 'a': auc_a, 'mAUC': mauc}


def pool2d(maxpool_input, kernel_size, stride, padding=0):
    maxpool_input = np.pad(maxpool_input, padding, mode='constant')
    # Window view of A
    output_shape = ((maxpool_input.shape[0] - kernel_size) // stride + 1,
                    (maxpool_input.shape[1] - kernel_size) // stride + 1)
    
    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    strides_w = (stride*maxpool_input.strides[0], stride*maxpool_input.strides[1],
    maxpool_input.strides[0], maxpool_input.strides[1])
    
    a_w = np.lib.stride_tricks.as_strided(maxpool_input, shape_w, strides_w)

    return a_w.max(axis=(2, 3))


def max_pooling(x, kernel_size, stride, padding):
    x1 = x.copy()
    a = x[0, 0, :, :]
    b = x[1, 0, :, :]
    x1[0, 0, :, :] = pool2d(a, kernel_size, stride, padding)
    x1[1, 0, :, :] = pool2d(b, kernel_size, stride, padding)
    
    return x1


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby geo_points """
    assert (nms_radius >= 0)

    def max_pool(x):
        return max_pooling(
            x, kernel_size=nms_radius * 2 + 1, stride=1,
            padding=nms_radius)

    zeros = np.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.astype(np.float32)) > 0
        supp_scores = np.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return np.where(max_mask, scores, zeros)


def grid_sample(sample_input, grid):
    n_grid, c_grid, h_in, w_in = sample_input.shape
    n_grid, h_out, w_out, _ = grid.shape
    output = np.random.random((n_grid, c_grid, h_out, w_out))
    for i in range(n_grid):
        for j in range(c_grid):
            for k in range(h_out):
                for l in range(w_out):
                    x, y = grid[i][k][l][0], grid[i][k][l][1]
                    param = [0.0, 0.0]
                    param[0] = (w_in - 1) * (x + 1) / 2
                    param[1] = (h_in - 1) * (y + 1) / 2
                    x1 = int(param[0] + 1)
                    x0 = x1 - 1
                    y1 = int(param[1] + 1)
                    y0 = y1 - 1
                    param[0] = abs(param[0] - x0)
                    param[1] = abs(param[1] - y0)
                    left_top_value, left_bottom_value, right_top_value, right_bottom_value = 0, 0, 0, 0
                    if 0 <= x0 < w_in and 0 <= y0 < h_in:
                        left_top_value = sample_input[i][j][y0][x0]
                    if 0 <= x1 < w_in and 0 <= y0 < h_in:
                        right_top_value = sample_input[i][j][y0][x1]
                    if 0 <= x0 < w_in and 0 <= y1 < h_in:
                        left_bottom_value = sample_input[i][j][y1][x0]
                    if 0 <= x1 < w_in and 0 <= y1 < h_in:
                        right_bottom_value = sample_input[i][j][y1][x1]
                    left_top = left_top_value * (1 - param[0]) * (1 - param[1])
                    left_bottom = left_bottom_value * (1 - param[0]) * param[1]
                    right_top = right_top_value * param[0] * (1 - param[1])
                    right_bottom = right_bottom_value * param[0] * param[1]
                    result = left_bottom + left_top + right_bottom + right_top
                    output[i][j][k][l] = result
    return output


def normalize(x, p, dim):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    x_norm = np.linalg.norm(x, ord=p, axis = dim, keepdims = True)
    x = np.divide(x, x_norm + 1e-6)
 
    return x


def cal_homography(query_path, refer_path, comp_model, query_is_image=False):
    goodmatch, cv_kpts_query, cv_kpts_refer, raw_query_image, raw_refer_image, comp_image_height, comp_image_width = \
    match(query_path, refer_path, comp_model, query_is_image=query_is_image)
    h_m = None
    comp_inliers_num_rate = 0

    if len(goodmatch) >= 4:
        src_pts = [cv_kpts_query[m.queryIdx].pt for m in goodmatch]
        src_pts = np.float32(src_pts).reshape(-1, 1, 2)
        dst_pts = [cv_kpts_refer[m.trainIdx].pt for m in goodmatch]
        dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

        h_m, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)

        goodmatch = np.array(goodmatch)[mask.ravel() == 1]
        comp_inliers_num_rate = mask.sum() / len(mask.ravel())
    return [h_m, comp_inliers_num_rate, raw_query_image, raw_refer_image, comp_image_height, comp_image_width]


def sample_keypoint_desc(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b_s, c_s, h_s, w_s = descriptors.shape
    keypoints = keypoints.copy().astype(np.float32)

    keypoints /= np.array([w_s * s - 1, h_s * s - 1]).astype(np.float32)[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)

    descriptors = grid_sample(
        descriptors, keypoints.reshape(b_s, 1, -1, 2))

    descriptors = normalize(
        descriptors.reshape(b_s, c_s, -1), p=2, dim=1)
    return descriptors


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def evaluate(query_path, refer_path, infer_model, query_is_image=False):
    raw_tensor_query , raw_tensor_refer, eval_image_height, eval_image_width =\
    image_read(query_path, refer_path, query_is_image=query_is_image)
    tensor_query = transform(raw_tensor_query)
    tensor_refer = transform(raw_tensor_refer)
    inputs  = np.concatenate((np.expand_dims(tensor_query, axis=0), np.expand_dims(tensor_refer, axis=0)), axis=0)
    inputs = inputs.astype(np.float32)
    inputs_tensor = sdk.Tensor(inputs)
    inputs_tensor.to_device(0)
    inputs_tensor_list = [inputs_tensor]
    outputs = infer_model.infer(inputs_tensor_list)
    outputs[0].to_host()
    outputs[1].to_host()
    infer_data_detector = outputs[0]
    infer_data_descriptor = outputs[1]
    infer_data_detector = np.array(infer_data_detector)
    infer_data_descriptor = np.array(infer_data_descriptor)

    # get the infer result
    detector_pred = np.reshape(infer_data_detector, (2, 1, 768, 768))
    descriptor_pred = np.reshape(infer_data_descriptor, (2, 256, 94, 94))

    scores = simple_nms(detector_pred, nms_radius=nms_size)
    _, _, h_detector, w_detector = detector_pred.shape
    scores = scores.reshape(-1, h_detector, w_detector)

    keypoints = [np.transpose(np.nonzero(s > nms_thresh)) for s in scores]

    scores = [s[tuple(np.array(k).T.tolist())] for s, k in zip(scores, keypoints)]

    # Discard keypoints near the image borders
    keypoints, scores = list(zip(*[
        remove_borders(np.array(k), np.array(s), 4, h_detector, w_detector)
        for k, s in zip(keypoints, scores)]))

    keypoints = [np.flip(k, [1]).astype(np.float32) for k in keypoints]

    descriptors = [sample_keypoint_desc(k[None], d[None], 8)[0]
                    for k, d in zip(keypoints, descriptor_pred)]
    keypoints = [k for k in keypoints]
    return [keypoints, descriptors, raw_tensor_query, raw_tensor_refer, eval_image_height, eval_image_width]


def match(query_path, refer_path, match_model, query_is_image=False):
    keypoints, descriptors, raw_query_image, raw_refer_image, match_image_height, match_image_width =\
    evaluate(query_path, refer_path, match_model, query_is_image=query_is_image)
    query_keypoints, refer_keypoints = keypoints[0], keypoints[1]
    query_desc, refer_desc = descriptors[0].T.astype(np.float32), descriptors[1].T.astype(np.float32)

    # mapping keypoints to scaled keypoints
    cv_kpts_query = [cv2.KeyPoint(int(i[0] / model_image_width * match_image_width),
                                    int(i[1] / model_image_height * match_image_height), 30)
                        for i in query_keypoints]
    cv_kpts_refer = [cv2.KeyPoint(int(i[0] / model_image_width * match_image_width),
                                    int(i[1] / model_image_height * match_image_height), 30)
                        for i in refer_keypoints]

    good_match = []
    status = []
    matches = []
    knn_matcher = cv2.BFMatcher(cv2.NORM_L2)
    try:
        matches = knn_matcher.knnMatch(query_desc, refer_desc, k=2)
        for m, n in matches:
            if m.distance < knn_thresh * n.distance:
                good_match.append(m)
                status.append(True)
            else:
                status.append(False)
    except Exception:
        pass
    return [good_match, cv_kpts_query, cv_kpts_refer, raw_query_image,
    raw_refer_image, match_image_height, match_image_width]


if __name__ == '__main__':
    # init stream manager
    FILEPATH = "./model/SuperRetina.om"       # om path
    DEVICEID = 0                             # device id
    model = sdk.model(FILEPATH, DEVICEID)

    # set stream name and device
    STREAM_NAME = b'superretina'
    IN_PLUGIN_ID = 0
    DATA_PATH = './data/'

    TESTSET = 'FIRE'

    if not os.path.isdir(DATA_PATH):
        print("data path don't exit.")
        exit()

    gt_dir = os.path.join(DATA_PATH, TESTSET, 'Ground Truth')
    im_dir = os.path.join(DATA_PATH, TESTSET, 'Images')
    OUT_PATH = '../result'
    matches_pair = [x for x in os.listdir(gt_dir) if x.endswith('.txt')
                and not x.endswith('P37_1_2.txt')]

    FAILED = 0
    INACCURATE = 0
    MAE = 0
    MEE = 0

    matches_pair.sort()
    BIGNUM = 1e6
    good_number = []
    IMAGENUM = 0
    # category: S, P, A, corresponding to Easy, Hard, Mod in paper
    auc_record = dict([(category, []) for category in ['S', 'P', 'A']])

    for read_file in tqdm(matches_pair):
        ground_file = os.path.join(gt_dir, read_file)
        name = read_file.replace('.txt', '')

        category = name.split('_')[2][0]

        name_rf = name.split('_')[2] + '_' + name.split('_')[3]
        name_qr = name.split('_')[2] + '_' + name.split('_')[4]

        query_im = os.path.join(im_dir, name_qr + '.jpg')
        refer_im = os.path.join(im_dir, name_rf + '.jpg')
        H_m1, inliers_number, query_image, _, image_height, image_width =\
            cal_homography(query_im, refer_im, model)
        HM2 = None
        if use_matching_trick:
            if H_m1 is not None:
                h, w = image_height, image_width
                query_align_one = cv2.warpPerspective(query_image, H_m1, (h, w), borderMode=cv2.BORDER_CONSTANT,
                                                borderValue=(0))
                query_align_one = query_align_one.astype(float)
                query_align_one /= 255.
                HM2, inliers_number, _, _, image_height, image_width =\
                cal_homography(query_align_one, refer_im, model, query_is_image=True)

        good_number.append(inliers_number)
        IMAGENUM += 1

        if inliers_number < 1e-6:
            FAILED += 1
            AVGDIST = BIGNUM
        else:
            ground_points = np.loadtxt(ground_file)
            origin = np.zeros([len(ground_points), 2])
            destination = np.zeros([len(ground_points), 2])
            origin[:, 0] = ground_points[:, 2]
            origin[:, 1] = ground_points[:, 3]
            destination[:, 0] = ground_points[:, 0]
            destination[:, 1] = ground_points[:, 1]
            destination_pred = cv2.perspectiveTransform(origin.reshape(-1, 1, 2), H_m1)
            if HM2 is not None:
                destination_pred = cv2.perspectiveTransform(destination_pred.reshape(-1, 1, 2), HM2)

            destination_pred = destination_pred.squeeze()

            difference = (destination - destination_pred) ** 2
            difference = np.sqrt(difference[:, 0] + difference[:, 1])
            AVGDIST = difference.mean()
            
            MAE = difference.max()
            MEE = np.median(difference)
            if MAE > 50 or MEE > 20:
                INACCURATE += 1
        auc_record[category].append(AVGDIST)
    
    print('----------------------------------------------------------')
    print(f"fail:{'%.2f' % (100*FAILED/IMAGENUM)}%, inaccurate:{'%.2f' % (100*INACCURATE/IMAGENUM)}%, "
        f"acceptable:{'%.2f' % (100*(IMAGENUM-INACCURATE-FAILED)/IMAGENUM)}%")

    print('----------------------------------------------------------')

    auc = cal_auc(auc_record['S'], auc_record['P'], auc_record['A'])
    print('S: %.3f, P: %.3f, A: %.3f, mAUC: %.3f' % (auc.get('s'), auc.get('p'), auc.get('a'), auc.get('mAUC')))
   