#!/usr/bin/env python
# coding=utf-8

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


import json
import os
import cv2
import numpy as np
from PIL import Image
import torch
import stat
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxProtobufIn, InProtobufVector, StringVector
from util.misc import fill_hole, regularize_sin_cos
from util.detection import TextDetector
from util.misc import to_device, mkdirs, rescale_result
from util.config import config as cfg
from util.visualize import visualize_detection


def norm(image,mean,std):
    image = image.astype(np.float32)
    image /= 255.0
    image -= mean
    image /= std
    return image

def resize(image,size):
    h, w, _ = image.shape
    image = cv2.resize(image, (size,size))
    scales = np.array([size / w, size / h])
    return image

if __name__ == '__main__':
    steam_manager_api = StreamManagerApi()
    
    ret = steam_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    
    
    MODES = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open("./t.pipeline", os.O_RDONLY, MODES), 'rb') as f:
        pipeline_str = f.read()
    ret = steam_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)
    IMAGE_PATH = './img1.jpg'
    image = Image.open(IMAGE_PATH)
    image = np.array(image)
    H, W, _ = image.shape
    image=resize(image,cfg.input_size)
    image=norm(image,np.array(means),np.array(stds))
    image=image.transpose(2, 0, 1)
    visionList = MxpiDataType.MxpiVisionList()
    visionVec = visionList.visionVec.add()

    visionInfo = visionVec.visionInfo
    visionInfo.width = image.shape[1]
    visionInfo.height = image.shape[0]
    visionInfo.widthAligned = image.shape[1]
    visionInfo.heightAligned = image.shape[0]

    visionData = visionVec.visionData
    visionData.dataStr = image.tobytes()
    visionData.deviceId = 0
    visionData.memType = 0
    visionData.dataSize = len(image)

    KEY0 = b"appsrc0"

    protobufVec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = KEY0
    protobuf.type = b"MxTools.MxpiVisionList"
    protobuf.protobuf = visionList.SerializeToString()
    protobufVec.push_back(protobuf)
    STEAMNAME = b'detection'
    INPLUGINID = 0
    uniqueId = steam_manager_api.SendProtobuf(STEAMNAME, INPLUGINID, protobufVec)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()
    keys = [b"mxpi_tensorinfer0"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)
    infer = steam_manager_api.GetResult(STEAMNAME, b'appsink0', keyVec)
    if(infer.metadataVec.size() == 0):
        print("Get no data from stream !")
        exit()
    infer_result = infer.metadataVec[0]
    if infer_result.errorCode != 0:
        print("GetResult error. errorCode=%d , errMsg=%s" % (infer_result.errorCode, infer_result.errMsg))
        exit()
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result.serializedMetadata)
    pred = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
    pred_array=pred.reshape(1,7,512,512)
    tr_pred=pred_array[:,0:2,:,:].reshape(2,512,512)
    tcl_pred=pred_array[:,2:4,:,:].reshape(2,512,512)
    sin_pred=pred_array[:,4,:,:].reshape(512,512)
    cos_pred=pred_array[:,5,:,:].reshape(512,512)
    radii_pred=pred_array[:,6,:,:].reshape(512,512)
    tr_pred_tensor= torch.from_numpy(tr_pred)
    tcl_pred_tensor= torch.from_numpy(tcl_pred)
    tr_pred = tr_pred_tensor.softmax(dim=0).data.cpu().numpy()
    tcl_pred = tcl_pred_tensor.softmax(dim=0).data.cpu().numpy()
    td=TextDetector(cfg.tr_thresh,cfg.tcl_thresh)
    contours = td.detect_contours(image, tr_pred, tcl_pred, sin_pred, cos_pred, radii_pred)
    output = {
        'image': image,
        'tr': tr_pred,
        'tcl': tcl_pred,
        'sin': sin_pred,
        'cos': cos_pred,
        'radii': radii_pred
    }

    
    tr_pred, tcl_pred = output['tr'], output['tcl']
    
    img_show = image.transpose(1,2,0)
    img_show = ((img_show * stds + means) * 255).astype(np.uint8)
    img_show, contours = rescale_result(img_show, contours, H, W)
    vis_dir = "result.jpg"
    pred_vis = visualize_detection(img_show, contours)
    mkdirs(vis_dir)
    cv2.imwrite(vis_dir, pred_vis)
    steam_manager_api.DestroyAllStreams()