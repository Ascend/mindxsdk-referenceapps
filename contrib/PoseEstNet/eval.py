#!/usr/bin/env python
# coding=utf-8

"""
Copyright(C) Huawei Technologies Co.,Ltd. 2012-2022 All rights reserved.

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

import argparse
import os
import stat
import csv
import math
import numpy as np

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

YUV_BYTES_NU = 3
YUV_BYTES_DE = 2

POSEESTNET_STREAM_NAME = b'PoseEstNetProcess'
IN_PLUGIN_ID = 0


def initialize_stream():
    """
    Initialize stream
    :arg:
        None
    :return:
        Stream api
    """
    stream_api = StreamManagerApi()
    stream_state = stream_api.InitManager()
    if stream_state != 0:
        error_message = "Failed to init Stream manager, stream_state=%s" % str(stream_state)
        print(error_message)
        exit()

    # creating stream based on json strings in the pipeline file: 'ReID.pipeline'
    with open("pipeline/eval_PoseEstNet.pipeline", 'rb') as f:
        pipeline = f.read().replace(b'\r', b'').replace(b'\n', b'')
    pipeline_string = pipeline

    stream_state = stream_api.CreateMultipleStreams(pipeline_string)
    if stream_state != 0:
        error_message = "Failed to create Stream, stream_state=%s" % str(stream_state)
        print(error_message)
        exit()

    return stream_api


def get_result(input_path, label_path, stream_api):
    # constructing the results returned by the stream
    plugin_names = [b"mxpi_postprocess1"]
    plugin_name_vector = StringVector()
    for key in plugin_names:
        plugin_name_vector.push_back(key)

    # check the file
    if os.path.exists(input_path) != 1:
        error_message = 'The file of input images does not exist.'
        print(error_message)
        exit()
    if len(os.listdir(input_path)) == 0:
        error_message = 'The file of input images is empty.'
        print(error_message)
        exit()

    label_csv = open(label_path)
    reader = csv.reader(label_csv, delimiter=',')
    hash_annot = {}
    for row in reader:
        img_name = row[0]
        width = int(row[1])
        height = int(row[2])
        joints = []
        for j in range(36):
            joint = [int(row[j * 3 + 3]), int(row[j * 3 + 4]), int(row[j * 3 + 5])]
            joints.append(joint)
        hash_annot[img_name] = (width, height, joints)
    all_preds = np.zeros((len(hash_annot), 36, 3), dtype=np.float32)
    batch_count = 0
    idx = 0
    image_names = []

    for k in sorted(hash_annot.keys()):
        image_name = k
        image_names.append(image_name)
        if not image_name.lower().endswith((".jpg")):
            print('Input image only support jpg')
            exit()
        img_path = os.path.join(input_path, image_name)
        query_data_input = MxDataInput()
        with open(img_path, 'rb') as f:
            query_data_input.data = f.read()

        # send the prepared data to the stream
        unique_id = stream_api.SendData(POSEESTNET_STREAM_NAME, IN_PLUGIN_ID, query_data_input)
        if unique_id < 0:
            error_message = 'Failed to send data to queryImageProcess stream.'
            print(error_message)
            exit()

        # get infer result
        infer_result = stream_api.GetProtobuf(POSEESTNET_STREAM_NAME, IN_PLUGIN_ID, plugin_name_vector)
        # checking whether the infer results is valid or not
        if infer_result.size() == 0:
            error_message = 'Unable to get effective infer results, please check the stream log for details'
            print(error_message)
            exit()
        if infer_result[0].errorCode != 0:
            error_message = "GetProtobuf error. errorCode=%d, error_message=%s" % (infer_result[0].errorCode,
                                                                                 infer_result[0].messageName)
            print(error_message)
            exit()

        # get the output
        object_list = MxpiDataType.MxpiObjectList()
        object_list.ParseFromString(infer_result[0].messageBuf)
        for index in range(len(object_list.objectVec)):
            x = object_list.objectVec[index].x0
            y = object_list.objectVec[index].y0
            vision = object_list.objectVec[index].x1
            all_preds[idx + batch_count, index, 0] = x
            all_preds[idx + batch_count, index, 1] = y
            all_preds[idx + batch_count, index, 2] = vision

        if batch_count == 31:
            print(f'-------- Test: [{int((idx+1)/32 + 1)}/{int(len(hash_annot)/32)}] ---------')
            idx += batch_count + 1
            batch_count = 0
        else:
            batch_count += 1

    # output pose in CSV format
    preds_length = len(all_preds)
    output_pose = os.open('output_eval/pose_test.csv', os.O_RDWR | os.O_CREAT, stat.S_IRWXU | stat.S_IRGRP)
    for p in range(preds_length):
        os.write(output_pose, ("%s," % (image_names[p])).encode())
        key_point_num = len(all_preds[p])
        for k in range(key_point_num-1):
            os.write(output_pose, ("%.3f,%.3f,%.3f," % (all_preds[p][k][0],
                                                        all_preds[p][k][1],
                                                        all_preds[p][k][2])).encode())
        os.write(output_pose, ("%.3f,%.3f,%.3f\n" % (all_preds[p][len(all_preds[p])-1][0],
                                                     all_preds[p][len(all_preds[p])-1][1],
                                                     all_preds[p][len(all_preds[p])-1][2])).encode())
    os.close(output_pose)


def evaluate(label_path):
    sc_bias = 0.25
    threshold = 0.5

    preds_read = []
    with open('output_eval/pose_test.csv') as annot_file:
        reader = csv.reader(annot_file, delimiter=',')
        for row in reader:
            joints = []
            for j in range(36):
                joint = [float(row[j * 3 + 1]), float(row[j * 3 + 2]), float(row[j * 3 + 3])]
                joints.append(joint)
            preds_read.append(joints)
    gts = []
    viss = []
    area_sqrts = []
    with open(label_path) as annot_file:
        reader = csv.reader(annot_file, delimiter=',')
        for row in reader:
            joints = []
            vis = []
            top_lft = btm_rgt = [int(row[3]), int(row[4])]
            for j in range(36):
                joint = [int(row[j * 3 + 3]), int(row[j * 3 + 4]), int(row[j * 3 + 5])]
                joints.append(joint)
                vis.append(joint[2])
                if joint[0] < top_lft[0]:
                    top_lft[0] = joint[0]
                if joint[1] < top_lft[1]:
                    top_lft[1] = joint[1]
                if joint[0] > btm_rgt[0]:
                    btm_rgt[0] = joint[0]
                if joint[1] > btm_rgt[1]:
                    btm_rgt[1] = joint[1]
            gts.append(joints)
            viss.append(vis)
            area_sqrts.append(math.sqrt((btm_rgt[0] - top_lft[0] + 1) * (btm_rgt[1] - top_lft[1] + 1)))

    jnt_visible = np.array(viss, dtype=int)
    jnt_visible = np.transpose(jnt_visible)
    pos_pred_src = np.transpose(preds_read, [1, 2, 0])
    pos_gt_src = np.transpose(gts, [1, 2, 0])
    uv_error = pos_pred_src - pos_gt_src
    uv_err = np.linalg.norm(uv_error, axis=1)
    area_sqrts = np.linalg.norm(area_sqrts, axis=0)
    area_sqrts *= sc_bias
    scale = np.multiply(area_sqrts, np.ones((len(uv_err), 1)))
    scaled_uv_err = np.divide(uv_err, scale)
    scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
    jnt_count = np.sum(jnt_visible, axis=1)
    less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                      jnt_visible)
    pckh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)

    # save
    rng = np.arange(0, 0.5 + 0.01, 0.01)
    pck_all = np.zeros((len(rng), 36))

    length_rng = len(rng)
    for r in range(length_rng):
        threshold = rng[r]
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        pck_all[r, :] = np.divide(100. * np.sum(less_than_threshold, axis=1),
                                 jnt_count)

    pckh = np.ma.array(pckh, mask=False)
    jnt_count = np.ma.array(jnt_count, mask=False)
    jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

    name_value = {
        'Wheel': (1.0 / 4.0) * (pckh[0] + pckh[1] + pckh[18] + pckh[19]),
        'Fender': (1.0 / 16.0) * (pckh[2] + pckh[3] + pckh[4] + pckh[5] + pckh[6] + pckh[7] + pckh[8] +
                                  pckh[9] + pckh[20] + pckh[21] + pckh[22] + pckh[23] + pckh[24] +
                                  pckh[25] + pckh[26] + pckh[27]),
        'Back': (1.0 / 4.0) * (pckh[10] + pckh[11] + pckh[28] + pckh[29]),
        'Front': (1.0 / 4.0) * (pckh[16] + pckh[17] + pckh[34] + pckh[35]),
        'WindshieldBack': (1.0 / 4.0) * (pckh[12] + pckh[13] + pckh[30] + pckh[31]),
        'WindshieldFront': (1.0 / 4.0) * (pckh[14] + pckh[15] + pckh[32] + pckh[33]),
        'Mean': np.sum(pckh * jnt_ratio),
        'Mean@0.1': np.sum(pck_all[11, :] * jnt_ratio)
    }

    _print_name_value(name_value, 'PoseEstNet')


def _print_name_value(name_value, full_arch_name):
    ''' print accuracy '''
    names = name_value.keys()
    values = name_value.values()
    print(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    print(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputPath', type=str, default='data_eval/images', help="Query File Path")
    parser.add_argument('--labelPath', type=str, default='data_eval/labels/label_test.csv', help="Gallery File Path")
    opt = parser.parse_args()
    streamManagerApi = initialize_stream()
    get_result(opt.inputPath, opt.labelPath, streamManagerApi)
    evaluate(opt.labelPath)