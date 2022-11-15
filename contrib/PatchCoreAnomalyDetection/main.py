# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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


import argparse
# import StreamManagerApi.py
import datetime
import json
import time

import MxpiDataType_pb2 as MxpiDataType
import cv2
from StreamManagerApi import *
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
import numpy as np

import os
from PIL import Image
from sklearn.metrics import roc_auc_score
from utils import FaissNN, preprocess, norm, PatchMaker, RescaleSegmentor
import yaml

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
cfg = open("config.yaml", 'r', encoding='utf-8')
data_dict = yaml.safe_load(cfg)
cfg.close()

cfg = open("config.yaml", 'r', encoding='utf-8')
data_dict = yaml.safe_load(cfg)
cfg.close()

parser = argparse.ArgumentParser(description='main')

parser.add_argument('--data', "-d", type=str, default="bottle")
args = parser.parse_args()
category = args.data
RESIZE_IMG = data_dict[category]["resize"]
CROP_IMG = data_dict[category]["imagesize"]
model = data_dict[category]["backbone"]
feature_layer = data_dict[category]["layer"]
scales = [8, 16]
emb_0 = int(CROP_IMG / scales[0])
emb_1 = int(CROP_IMG / scales[1])
if feature_layer == "layer2":
    emb_area = int(emb_0 * emb_0)
    channel_cnt = 512
else:
    emb_area = int(emb_1 * emb_1)
    channel_cnt = 1024
    emb_0 = emb_1
anomaly_segmentor = RescaleSegmentor(
    target_size=(RESIZE_IMG, RESIZE_IMG)
)

os.makedirs(f"segmention/{category}", exist_ok=True)
if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    pipelineStr = {

        "classification+detection": {
            "stream_config": {
                "deviceId": "2"
            },
            "mxpi_tensorinfer0": {
                "props": {
                    "dataSource": "appsrc0",
                    "modelPath": f"models/{model}_{CROP_IMG}_{feature_layer}.om"
                },
                "factory": "mxpi_tensorinfer",
                "next": "appsink0"
            },
            "appsrc0": {
                "props": {
                    "blocksize": "409600"
                },
                "factory": "appsrc",
                "next": "mxpi_tensorinfer0"
            },
            "appsink0": {

                "factory": "appsink"
            }
        }
    }

    # create streams by pipeline config file
    pipelineStr = json.dumps(pipelineStr).encode()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    classpath = f"mvtec/{category}/test/"
    anomaly_types = os.listdir(classpath)

    data_tuple = []

    for anomaly in anomaly_types:
        os.makedirs(f"segmention/{category}/{anomaly}", exist_ok=True)
        images = os.path.join(classpath, anomaly)
        for img_path in os.listdir(images):
            if img_path.endswith(".jpg") or img_path.endswith("pre.jpg"):
                continue
            img_path = os.path.join(images, img_path)
            data_tuple.append([img_path, anomaly])

    scores = []
    labels = []
    scorestopK10 = []
    images_cnt = len(data_tuple)
    cnt = 0
    all_time = 0
    patch_maker = PatchMaker(patchsize=3)
    nn_method = FaissNN(4)
    faiss_path = f"faiss-index-precision/{category}/nnscorer_search_index.faiss"
    nn_method.load(faiss_path)
    imagelevel_nn = lambda query: nn_method.run(
        1, query
    )

    for img in data_tuple:
        cnt += 1
        print(f"{cnt}/{images_cnt}")

        ori_img = cv2.imread(img[0])
        ori_img = np.transpose(ori_img, (2, 0, 1))[::-1]
        ori_img = np.transpose(ori_img, (1, 2, 0))

        ori_img = cv2.resize(ori_img, [RESIZE_IMG, RESIZE_IMG], interpolation=cv2.INTER_AREA)
        left = int((RESIZE_IMG - CROP_IMG) / 2)
        right = left + CROP_IMG
        ori_img = ori_img[left:right, left:right, :]

        ori_img = (ori_img / 255. - IMAGENET_MEAN) / IMAGENET_STD
        ori_img = np.transpose(ori_img, (2, 0, 1))
        ori_img = np.ascontiguousarray(ori_img, dtype=np.float32)
        img[0] = img[0].split("/")[-1]
        img[0] = img[0].split(".")[0] + "_seg.jpg"

        start = time.time()
        tensor_pack_list = preprocess(ori_img)
        # send data to stream
        proto_buffer_in = MxProtobufIn()
        proto_buffer_in.key = b'appsrc0'
        proto_buffer_in.type = b'MxTools.MxpiTensorPackageList'
        proto_buffer_in.protobuf = tensor_pack_list.SerializeToString()

        proto_buffer_vec = InProtobufVector()
        proto_buffer_vec.push_back(proto_buffer_in)

        # Inputs data to a specified stream based on streamName.
        streamName = b'classification+detection'
        inPluginId = 0

        ret = streamManagerApi.SendProtobuf(streamName, inPluginId, proto_buffer_vec)
        if ret < 0:
            print("Failed to send data to stream.")
            exit()

        # Obtain the inference result by specifying streamName and uniqueId.
        keys = [b"mxpi_tensorinfer0"]
        keyVec = StringVector()
        for key in keys:
            keyVec.push_back(key)
        infer = streamManagerApi.GetResult(streamName, b'appsink0', keyVec)
        if (infer.metadataVec.size() == 0):
            print("Get no data from stream !")
            exit()
        print("result.metadata size: ", infer.metadataVec.size())
        infer_result = infer.metadataVec[0]
        if infer_result.errorCode != 0:
            print("GetResult error. errorCode=%d , errMsg=%s" % (infer_result.errorCode, infer_result.errMsg))
            exit()

        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result.serializedMetadata)

        pred = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)

        features = pred.reshape(1, channel_cnt, emb_0, emb_0).astype('float32')

        features = np.transpose(features, (0, 2, 3, 1))
        features = features.reshape(int(features.shape[1] * features.shape[2]), channel_cnt)

        features = np.ascontiguousarray(features)

        query_features = features.reshape(len(features), -1)

        query_distances, query_nns = imagelevel_nn(query_features)
        patch_scores = image_scores = np.mean(query_distances, axis=-1)

        topK = 10
        index = np.argsort(patch_scores)[::-1][0:topK]
        sum = 0
        for idx in index:
            sum += patch_scores[idx.item()]
        avg = sum / topK
        scorestopK10.append(avg.item())

        image_scores = patch_maker.unpatch_scores(
            image_scores, batchsize=1
        )
        image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
        image_scores = patch_maker.score(image_scores).item()
        scores.append(image_scores)
        labels.append(1 if img[1] != "good" else 0)
        end = time.time()
        step_time = end - start
        all_time += step_time
        patch_scores = patch_scores.reshape(1, emb_0, emb_0)
        tmp = anomaly_segmentor.convert_to_segmentation(patch_scores)
        segmentations = np.array(tmp)
        min_scores = (
            segmentations.reshape(len(segmentations), -1)
            .min(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        max_scores = (
            segmentations.reshape(len(segmentations), -1)
            .max(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        segmentations = (segmentations - min_scores) / (max_scores - min_scores)
        segmentations = np.mean(segmentations, axis=0)
        segmentations = np.transpose(segmentations, (1, 2, 0))
        segmentations = segmentations * 255.
        segmentations = segmentations.astype(np.uint8)
        heat_img = cv2.applyColorMap(segmentations, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(f"segmention/{category}/{img[1]}", img[0]), heat_img)
        # img = Image.fromarray(segmentations)
        # img.save("1.jpg")
        print("infer time: {}s".format(step_time))
        RescaleSegmentor()
    avg_time = all_time / cnt
    print("avg infer time: {}s".format(avg_time))

    scores = np.array(scores, dtype=np.float32)
    min_scores = scores.min(axis=-1).reshape(-1, 1)
    max_scores = scores.max(axis=-1).reshape(-1, 1)
    scores = (scores - min_scores) / (max_scores - min_scores)
    scores = np.mean(scores, axis=0)

    scorestopK10 = np.array(scorestopK10, dtype=np.float32)

    scorestopK10 = norm(scorestopK10)

    auroc = roc_auc_score(np.array(labels), scores)
    auroc_topK10 = roc_auc_score(np.array(labels), scorestopK10)
    print(f"auroc:{auroc}")
    print(f"auroc_topK10:{auroc_topK10}")
    streamManagerApi.DestroyAllStreams()
