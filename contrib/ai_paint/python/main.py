#!/usr/bin/env python
# coding=utf-8

"""
Copyright(C) Huawei Technologies Co.,Ltd. 2012-2021 All rights reserved.

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
# USE LF FORMAT!
from StreamManagerApi import StreamManagerApi, MxProtobufIn, InProtobufVector, StringVector
import json
import os
import time
import configparser
import cv2
import numpy as np
import MxpiDataType_pb2 as MxpiDataType

vocab = {"sky": 1, "sand": 2, "sea": 3, "mountain": 4, "rock": 5, "earth": 6, "tree": 7, "water": 8,
         "land": 9, "grass": 10, "path": 11, "dirt": 12, "river": 13, "hill": 14, "filed": 15, "lake": 16}
VOC_CLASS_NUM = 17
MAX_OBJ_PER_IMG = 9
LAYOUT_HW_LEN = 256
RES_HE_LEN = 448
IMG_CHN_NUM = 3
IMG_CHN_MAX = 255
RES_RESIZE_FIX = 0.5

def gen_coarse_layout(objs, boxes, attributes, obj_valid_inds, layout_size=LAYOUT_HW_LEN, num_classes=VOC_CLASS_NUM):
    height, width = layout_size, layout_size
    layout = np.zeros((1, height, width, num_classes), dtype=float)
    for objs_item, box, attributes_item, obj_valid_inds_item in zip(objs, boxes, attributes, obj_valid_inds):
        if obj_valid_inds_item == 0:
            break
        if box[0] > 1 or box[0] < 0:
            print("error boxes x value with float in [0,1]! input: ", box[0])
            exit()
        if box[1] > 1 or box[1] < 0:
            print("error boxes y value with float in [0,1]! input: ", box[1])
            exit()
        if attributes_item > 9 or attributes_item < 0:
            print("error size_att value with int in [1,9]! input: ", attributes_item)
            exit()
        x_c, y_c = width * box[0], height * box[1]
        obj_size = attributes_item
        w, h = width * float(obj_size) / 10, height * float(obj_size) / 10
        x0, y0, x1, y1 = int(x_c - w / 2), int(y_c - h / 2), int(x_c + w / 2), int(y_c + h / 2)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, width), min(y1, height)
        layout[:, y0:y1 - 1, x0:x1 - 1, int(objs_item)] = 1

    layout[:, :, :, 0] = 0
    return layout

def preprocess(net_param):
    # input param
    objects_str = net_param.get("net_param", "objects")
    objects = json.loads(objects_str)
    objects_num = len(objects)
    print("input object number:", objects_num)
    if objects_num > 9:
        print("objects limit ! max 9 input %d" % objects_num)
        exit()
    boxes_str = net_param.get("net_param", "boxes")
    boxes = json.loads(boxes_str)
    if len(boxes) != objects_num:
        print("boxes not pair object! input: %d" % len(boxes))
        exit()
    size_att_str = net_param.get("net_param", "size_att")
    size_att = json.loads(size_att_str)
    if len(size_att) != objects_num:
        print("size_att not pair object! input: %d" % len(size_att))
        exit()

    # gen np array
    num_objs = len(objects)
    objects = np.array([vocab[name] for name in objects])
    to_pad = MAX_OBJ_PER_IMG - num_objs
    objects = np.pad(objects, (0, to_pad), mode='constant').astype(np.int64)
    obj_valid_inds = np.array([1] * num_objs + [0] * to_pad)
    layout = np.array(gen_coarse_layout(objects, boxes, size_att, obj_valid_inds, layout_size=LAYOUT_HW_LEN, num_classes=VOC_CLASS_NUM),
                    dtype=np.float32)
    
    # gen tensor data
    mxpi_tensor_pack_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package_vec = mxpi_tensor_pack_list.tensorPackageVec.add()

    # add object data
    tensorVec_obj = tensor_package_vec.tensorVec.add()
    tensorVec_obj.memType = 1
    tensorVec_obj.deviceId = 0
    tensorVec_obj.tensorDataSize = int(MAX_OBJ_PER_IMG * 8) # obj * sizeof(int64)
    tensorVec_obj.tensorDataType = 9 # int64
    for i in objects.shape:
        tensorVec_obj.tensorShape.append(i)
    tensorVec_obj.dataStr = objects.tobytes()

    # add layout data
    tensorVec_lay = tensor_package_vec.tensorVec.add()
    tensorVec_lay.memType = 1
    tensorVec_lay.deviceId = 0
    tensorVec_lay.tensorDataSize = int(layout.shape[1] * layout.shape[2] * IMG_CHN_NUM * 4) # H * W * C * sizeof(float32)
    tensorVec_lay.tensorDataType = 0 # float32
    for i in layout.shape:
        tensorVec_lay.tensorShape.append(i)
    tensorVec_lay.dataStr = layout.tobytes()

    return mxpi_tensor_pack_list

def read_config(config_fname):
    curpath = os.path.dirname(os.path.realpath(__file__))
    cfgpath = os.path.join(curpath, config_fname)
    
    conf = configparser.ConfigParser()
    conf.read(cfgpath, encoding = "utf-8")

    return conf

if __name__ == '__main__':
    # init stream manager
    stream_manager = StreamManagerApi()
    ret = stream_manager.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("../pipeline/ai_paint.pipeline", 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # set stream name and device
    stream_name = b'ai_paint'
    in_plugin_id = 0
    config_file = 'net_config.ini'

    # send data to stream
    net_config = read_config(config_file)
    tensor_pack_list = preprocess(net_config)

    protobuf_in = MxProtobufIn()
    protobuf_in.key = b'appsrc0'
    protobuf_in.type = b'MxTools.MxpiTensorPackageList'
    protobuf_in.protobuf = tensor_pack_list.SerializeToString()

    protobuf_vec = InProtobufVector()
    protobuf_vec.push_back(protobuf_in)
    
    time_start = time.time()
    unique_id = stream_manager.SendProtobuf(stream_name, in_plugin_id, protobuf_vec)
    if unique_id < 0:
        print("Failed to send data to stream.")
        exit()

    # get inference result
    keys = [b"mxpi_tensorinfer0"]
    key_vec = StringVector()
    for key in keys:
        key_vec.push_back(key)

    infer_raw = stream_manager.GetResult(stream_name, b'appsink0', key_vec)
    print("result.metadata size: ", infer_raw.metadataVec.size())
    infer_result = infer_raw.metadataVec[0]

    if infer_result.errorCode != 0:
        print("GetResult error. errorCode=%d , errMsg=%s" % (
            infer_result.errorCode, infer_result.errMsg))
        exit()
    time_end = time.time()
    print('Time cost = %fms' % ((time_end - time_start) * 1000))

    # convert result
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result.serializedMetadata)
    print("tensorPackageVec size=%d, tensorPackageVec[0].tensorVec size=%d" % ( 
        len(result.tensorPackageVec), len(result.tensorPackageVec[0].tensorVec)))

    img1_rgb = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr
        , dtype = np.float32)
    print("raw output shape:", result.tensorPackageVec[0].tensorVec[0].tensorShape)
    img1_rgb.resize(LAYOUT_HW_LEN, LAYOUT_HW_LEN, IMG_CHN_NUM)

    img2_rgb = np.frombuffer(result.tensorPackageVec[0].tensorVec[1].dataStr
        , dtype = np.float32)
    img2_rgb.resize(RES_HE_LEN, RES_HE_LEN, IMG_CHN_NUM)
    img2_rgb = (img2_rgb * RES_RESIZE_FIX + RES_RESIZE_FIX) * IMG_CHN_MAX

    # save result image
    img1_bgr = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)
    img2_bgr = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite("../result/layoutMap.jpg", img1_bgr)
    cv2.imwrite("../result/resultImg.jpg", img2_bgr)
    
    # destroy streams
    stream_manager.DestroyAllStreams()