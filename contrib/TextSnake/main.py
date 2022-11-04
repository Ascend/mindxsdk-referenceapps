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
from misc import fill_hole, regularize_sin_cos
from textdetector import TextDetector
from misc import to_device, mkdirs, rescale_result
from config import config as cfg
from visualize import visualize_detection
from imagereader import Resize,Normalize

if __name__ == '__main__':
    steam_manager_api = StreamManagerApi()
    # init stream manager
    ret = steam_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    # input_shape = [416, 416]
    # create streams by pipeline config file
    MODES = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open("./t.pipeline", os.O_RDONLY, MODES), 'rb') as f:
        pipeline_str = f.read()
    ret = steam_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    IMAGE_PATH = './img1.jpg'
    image = Image.open(IMAGE_PATH)
    image = np.array(image)
    H, W, _ = image.shape
    rsz=Resize(cfg.input_size)
    image,_=rsz(image)
    norm=Normalize(cfg.means,cfg.stds)
    image,_=norm(image)
    # success,image_b = cv2.imencode(".jpg",image)
    image=image.transpose(2, 0, 1)
    # image_b= image_b.transpose(2, 0, 1)
    # img_bytes = image_b.tostring()
    # image=Image.fromarray(image)
    # image.save(IMAGE_PATH)
    # cv2.imwrite(IMAGE_PATH1, image)
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
    # 从流中取出对应插件的输出数据
    #     infer = streamManagerApi.GetResult(STREAMNAME, b'appsink0', keyVec)
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
    # tr_pred = pred[0:524288]#0-524287
    # tcl_pred = pred[524288:1048576]#524287-1048575
    # sin_pred = pred[1048576:1310720]
    # cos_pred = pred[1310720:1572864]
    # radii_pred = pred[1572864:1835008]
    # tr_pred= np.array(tr_pred).reshape(2,512,512)
    # tcl_pred= np.array(tcl_pred).reshape(2,512,512)
    # sin_pred= np.array(sin_pred).reshape(512,512)
    # cos_pred= np.array(cos_pred).reshape(512,512)
    # radii_pred= np.array(radii_pred).reshape(512,512)
    tr_pred_tensor= torch.from_numpy(tr_pred)
    tcl_pred_tensor= torch.from_numpy(tcl_pred)
    tr_pred = tr_pred_tensor.softmax(dim=0).data.cpu().numpy()
    tcl_pred = tcl_pred_tensor.softmax(dim=0).data.cpu().numpy()
    # if filename=="img95.jpg" or filename=="img565.jpg" or filename=="img992.jpg":
    #     print("image",image,image.dtype)
    #     print("tr",tr_pred,tr_pred.dtype)
    #     print("tcl",tcl_pred,tcl_pred.dtype)
    #     print("sin",sin_pred,sin_pred.dtype)
    #     print("cos",cos_pred,cos_pred.dtype)
    #     print("radii_pred",radii_pred,radii_pred.dtype)
    # image= torch.from_numpy(image)#后面用的image都是tensor
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

    #image = to_device(image)
    tr_pred, tcl_pred = output['tr'], output['tcl']
    # idx=0 下面把image后面的[idx]也删掉了 这里batchsize相当于1
    img_show = image.transpose(1,2,0)#.cpu().numpy()#问题：dataloader load过的东西全换成tensor了 后处理处理的全是tensor tensor的维度应该是3，512，512，但pipeline里处理的是512，512，3 所以出来的可能不是对应3，512，512的数据
    img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)
    img_show, contours = rescale_result(img_show, contours, H, W)
    vis_dir = "result.jpg"
    pred_vis = visualize_detection(img_show, contours)
    # print(pred_vis.shape)
    cv2.imwrite(vis_dir, pred_vis)






    # pred.resize(HEIGHT+1, WIDTH+1)
    # preds = np.zeros((HEIGHT, WIDTH))
    # for i in range(HEIGHT):
    #     for j in range(WIDTH):
    #         if(pred[i+1][j+1] < 0):
    #             preds[i][j] = 0
    #         elif(pred[i+1][j+1] > 1):
    #             preds[i][j] = DE_NORM
    #         else:
    #             preds[i][j] = pred[i+1][j+1] * DE_NORM
    # end_array = np.array(preds, dtype=int)
    # SAVE_PATH = './result.jpg'
    # img_resize = cv2.resize(end_array, (w, h), interpolation = cv2.INTER_NEAREST)
    # cv2.imwrite(SAVE_PATH, img_resize)
    # destroy streams
    steam_manager_api.DestroyAllStreams()