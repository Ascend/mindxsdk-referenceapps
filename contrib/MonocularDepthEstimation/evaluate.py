#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
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

import numpy as np

from depth_estimation.monocular_depth_estimation import depth_estimation
from util.util import bilinear_sampling

# test image and depth info path
test_image_set_path = 'test_set/image'
test_image_depth_info_path = 'test_set/depth_info'

# evaluate result save path
evaluate_result_path = "result/evaluate_result.txt"

# cache file config
ground_truth_cache_file = 'test_set/ground_truth.npy'
infer_result_cache_file = 'test_set/infer_result.npy'

# test set image size
test_image_height = '${测试集图片的高}'
test_image_width = '${测试集图片的宽}'

# thresholds for accuracy
threshold_1 = 1.25
threshold_2 = 1.25 ** 2
threshold_3 = 1.25 ** 3


def load_test_set(image_set_path, image_depth_info_path):
    """
    load test images and whose depth info
    :param image_set_path: path of images
    :param image_depth_info_path: path of depth info
    :return: binary data of images and depth info of images
    """
    # check input paths
    input_valid = True
    if os.path.exists(image_set_path) != 1:
        input_valid = False
        print('The image set path {} does not exist.'.format(image_set_path))

    if os.path.exists(image_depth_info_path) != 1:
        input_valid = False
        print('The image depth info path {} does not exist.'.format(image_depth_info_path))

    if not input_valid:
        print('input is invalid.')
        return None, None

    # get all image files
    image_files = os.listdir(image_set_path)
    # sort by file name
    image_files.sort(key=lambda x: int(x[:-4]))

    # load image data
    images_data = []
    for test_image_file in image_files:
        # open image file
        with open(os.path.join(image_set_path, test_image_file), 'rb') as image_file:
            # load image data
            images_data.append(image_file.read())

    if os.path.exists(ground_truth_cache_file) != 1:
        print('ground_truth.npy not exist, query all data.')

        # get all depth info files
        image_depth_info_files = os.listdir(image_depth_info_path)

        # sort by file name
        image_depth_info_files.sort(key=lambda x: int(x[:-4]))

        # counter
        index = 0
        images_depth_data = np.empty(shape=(0, test_image_height, test_image_width))
        for test_image_depth_info_file in image_depth_info_files:
            index += 1
            # load depth data
            depth_data = np.load(os.path.join(image_depth_info_path, test_image_depth_info_file))
            print('loaded {}-th image depth info {}'.format(index, test_image_depth_info_file))

            # Combined data
            images_depth_data = np.vstack([images_depth_data, [depth_data]])

        # save ground truth cache
        print('save ground_truth.npy ...')
        np.save('test_set/ground_truth.npy', images_depth_data)
    else:
        # load ground truth cache
        print('ground_truth.npy exist, load ground_truth.npy.')
        images_depth_data = np.load(ground_truth_cache_file)

    return images_data, images_depth_data


def calculate_error(ground_truth, predict):
    """
    calculate errors between ground truth and predict value
    :param ground_truth: ground truth
    :param predict: predict value
    :return: errors (Absolute relative error, Square relative error, Root mean square error, Log root mean square error, log 10 error and accuracy)
    """
    # calculate absolute relative error
    abs_rel_error = np.abs(ground_truth - predict) / ground_truth
    abs_rel_error = np.mean(abs_rel_error)

    # calculate square relative error
    sq_rel_error = ((ground_truth - predict) ** 2) / ground_truth
    sq_rel_error = np.mean(sq_rel_error)

    # calculate root mean square error
    rmse_error = (ground_truth - predict) ** 2
    rmse_error = np.sqrt(rmse_error.mean())

    # calculate log root mean square error
    log_rmse_error = (np.log(ground_truth) - np.log(predict)) ** 2
    log_rmse_error = np.sqrt(log_rmse_error.mean())

    # calculate log 10 error
    log_10_error = np.abs(np.log10(ground_truth) - np.log10(predict))
    log_10_error = log_10_error.mean()

    # calculate accuracy
    thresh = np.maximum((ground_truth / predict), (predict / ground_truth))
    a1 = (thresh < threshold_1).mean()
    a2 = (thresh < threshold_2).mean()
    a3 = (thresh < threshold_3).mean()

    return dict(abs_rel_error=abs_rel_error, sq_rel_error=sq_rel_error, rmse_error=rmse_error,
                log_rmse_error=log_rmse_error, log_10_error=log_10_error, accuracy=[a1, a2, a3])


if __name__ == '__main__':

    # load test images data and ground truth
    test_images_data, test_images_depth_data = load_test_set(test_image_set_path, test_image_depth_info_path)

    if test_images_data is None or test_images_depth_data is None:
        print('load test set ( {} and {} ) failed.'.format(test_image_set_path, test_image_depth_info_path))
        exit(1)

    # model infer
    if os.path.exists(infer_result_cache_file) != 1:
        print('infer_result.npy not exist, infer all data.')

        # AdaBins_nyu model estimation
        test_images_infer_depth_info, test_images_info = depth_estimation(test_images_data, is_batch=True)

        # check depth estimation result
        if test_images_infer_depth_info is None or test_images_info is None or \
                test_images_infer_depth_info.shape[0] < test_images_depth_data.shape[0] or \
                test_images_info.shape[0] < len(test_images_data):
            print('depth estimation error on test set.')
            exit(1)

        # Bilinear sampling to restore to the original size
        test_images_infer_depth_info = bilinear_sampling(test_images_infer_depth_info, test_image_width,
                                                         test_image_height)
        # save infer result cache
        print('save infer_result.npy')
        np.save(infer_result_cache_file, test_images_infer_depth_info)
    else:
        # load infer result cache
        print('infer_result.npy exist, load infer_result.npy.')
        test_images_infer_depth_info = np.load(infer_result_cache_file)

    # calculate errors
    errors = calculate_error(test_images_depth_data, test_images_infer_depth_info)

    # print evaluation result
    image_num = test_images_infer_depth_info.shape[0]
    print('absolute relative error on {} test images is {}'.format(image_num, errors['abs_rel_error']))
    print('square relative error on {} test images is {}'.format(image_num, errors['sq_rel_error']))
    print('root mean square error on {} test images is {}'.format(image_num, errors['rmse_error']))
    print('log root mean square error on {} test images is {}'.format(image_num, errors['log_rmse_error']))
    print('log 10 error on {} test images is {}'.format(image_num, errors['log_10_error']))
    print('accuracy on {} test images is {}'.format(image_num, errors['accuracy']))

    # save errors
    with open(evaluate_result_path, 'w') as f:
        f.writelines(str(errors))
