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
import math
import stat
import subprocess
import cv2
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from PIL import Image
import torch

from StreamManagerApi import StreamManagerApi, MxProtobufIn, InProtobufVector, StringVector

from util.misc import fill_hole, regularize_sin_cos
from util.detection import TextDetector
from util.misc import to_device, mkdirs, rescale_result
from util.config import config as cfg
from util.visualize import visualize_detection


def zerodimsoftmax(x):
    first = x[0, :, :].reshape(512, 512)
    second = x[1, :, :].reshape(512, 512)
    fexp = np.exp(first)
    sexp = np.exp(second)
    sumexp = fexp + sexp
    fxf=fexp / sumexp
    fxs=sexp / sumexp
    fx = np.zeros((2, 512, 512))
    fx[0, :, :] = fxf
    fx[1, :, :] = fxs
    return fx



def norm(image_n, mean, std):
    image_n = image_n.astype(np.float32)
    image_n /= 255.0
    image_n -= mean
    image_n /= std
    return image_n


def resize(image_r, size):
    h, w, _ = image_r.shape
    image_r = cv2.resize(image_r, (size, size))
    return image_r


def write_to_file(contours_w, file_path):
    with os.fdopen(os.open(file_path, os.O_WRONLY|os.O_CREAT, MODES), 'w') as f1:
        for cont in contours_w:
            cont = np.stack([cont[:, 1], cont[:, 0]], 1)
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)
            f1.write(cont + '\n')


if __name__ == '__main__':
    try:
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
        FILEPATH = "./data/total-text/"
        if os.path.exists(FILEPATH) != 1:
            print("The filepath does not exist !")
            exit()
        for filename in os.listdir(FILEPATH):
            image_path = FILEPATH + filename
            if image_path.split('.')[-1] != 'jpg':
                continue
            IMAGE_PATH = image_path
            image = Image.open(IMAGE_PATH)
            image = np.array(image)
            H, W, _ = image.shape
            image = resize(image, cfg.input_size)
            image = norm(image, np.array(means), np.array(stds))
            image = image.transpose(2, 0, 1)
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
            pred_array = pred.reshape(1, 7, 512, 512)
            tr_pred = pred_array[:, 0: 2, : , :].reshape(2, 512, 512)
            tcl_pred = pred_array[:, 2:4, :, :].reshape(2, 512, 512)
            sin_pred = pred_array[:, 4, :, :].reshape(512, 512)
            cos_pred = pred_array[:, 5, :, :].reshape(512, 512)
            radii_pred = pred_array[:, 6, :, :].reshape(512, 512)
            tr_pred = zerodimsoftmax(tr_pred)
            tcl_pred = zerodimsoftmax(tcl_pred)
            td = TextDetector(cfg.tr_thresh, cfg.tcl_thresh)
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
            img_show = image.transpose(1, 2, 0)
            img_show = ((img_show * stds + means) * 255).astype(np.uint8)
            img_show, contours = rescale_result(img_show, contours, H, W)
            mkdirs(cfg.output_dir+"/test")
            write_to_file(contours, os.path.join(cfg.output_dir+"/test", filename.replace('jpg', 'txt')))
            pred_vis = visualize_detection(img_show, contours)
            

        steam_manager_api.DestroyAllStreams()
        print('Computing DetEval in {}/{}'.format(cfg.output_dir, "test"))
        subprocess.call(['python3.7', './util/Deteval.py', "test", '--tr', '0.7', '--tp', '0.6'])
        subprocess.call(['python3.7', './util/Deteval.py', "test", '--tr', '0.8', '--tp', '0.4'])
        print('End.')
    except KeyError:
        print("get result dict failed!")
    except ValueError:
        print("Tensor shape not match,maybe use wrong image type!")
    except FileNotFoundError:
        print("input image not found!")