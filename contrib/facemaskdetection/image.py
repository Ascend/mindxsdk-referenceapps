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

import json
import os
import cv2
import time
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import (
    StreamManagerApi,
    MxDataInput,
    StringVector,
    InProtobufVector,
    MxProtobufIn,
)
from anchor_generator import generate_anchors
from anchor_decode import decode_bbox
from nms import single_class_non_max_suppression


def inference(
    image,
    conf_thresh=0.5,
    iou_thresh=0.4,
    target_shape=(260, 260),
    draw_result=True,
    show_result=True,
):
    """
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size.
    :param draw_result: whether to daw bounding box to the image.
    :param show_result: whether to display the image.
    :return:
    """
    image = np.copy(image)
    output_info = []
    height, width, _ = image.shape
    y_bboxes_output = ids
    y_cls_output = ids2

    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)
    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(
        y_bboxes,
        bbox_max_scores,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
    )

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(
                image,
                "%s: %.2f" % (id2class[class_id], conf),
                (xmin + 2, ymin - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
            )
        output_info.append([class_id, conf, xmin + 5.1, ymin + 5.1, xmax + 5.1, ymax +5.1])

    if show_result:
        cv2.imwrite("./my_result.jpg", image)
    return output_info


if __name__ == "__main__":
    streamManagerApi = StreamManagerApi()
    # init stream manager
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    # create streams by pipeline config file
    pipeline_path = b"main.pipeline"
    ret = streamManagerApi.CreateMultipleStreamsFromFile(pipeline_path)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    img_path = "image/test7.jpg"
    streamName = b"detection"
    inPluginId = 0
    dataInput = MxDataInput()
    with open(img_path, "rb") as f:
        dataInput.data = f.read()
    # Inputs data to a specified stream based on streamName.
    uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    key_vec = StringVector()
    key_vec.push_back(b"mxpi_tensorinfer0")
    # get inference result
    infer_result = streamManagerApi.GetProtobuf(streamName, inPluginId, key_vec)
    a = infer_result.size()
    if infer_result.size() == 0:
        print("infer_result is null")
        exit()
    if infer_result[0].errorCode != 0:
        print(
            "GetProtobuf error. errorCode=%d, errorMsg=%s"
            % (infer_result[0].errorCode, infer_result[0].data.decode())
        )

        exit()
    tensorList = MxpiDataType.MxpiTensorPackageList()
    tensorList.ParseFromString(infer_result[0].messageBuf)

    # print the infer result
    ids = np.frombuffer(
        tensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32
    )
    shape = tensorList.tensorPackageVec[0].tensorVec[0].tensorShape
    ids.resize(shape)

    ids2 = np.frombuffer(
        tensorList.tensorPackageVec[0].tensorVec[1].dataStr, dtype=np.float32
    )
    shape2 = tensorList.tensorPackageVec[0].tensorVec[1].tensorShape
    ids2.resize(shape2)

    feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
    anchor_sizes = [
        [0.04, 0.056],
        [0.08, 0.11],
        [0.16, 0.22],
        [0.32, 0.45],
        [0.64, 0.72],
    ]
    anchor_ratios = [[1, 0.62, 0.42]] * 5

    # generate anchors
    anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

    # for inference , the batch size is 1, the model output shape is [1, N, 4],
    # so we expand dim for anchors to [1, anchor_num, 4]
    anchors_exp = np.expand_dims(anchors, axis=0)

    id2class = {0: "Mask", 1: "NoMask"}

    # visionList = MxpiDataType.MxpiVisionList()
    # visionList.ParseFromString(infer_result[1].messageBuf)
    # visionData = visionList.visionVec[0].visionData.dataStr
    # visionInfo = visionList.visionVec[0].visionInfo

    # YUV_BYTES_NU = 3
    # YUV_BYTES_DE = 2
    # img_yuv = np.frombuffer(visionData, dtype=np.uint8)
    # img_yuv = img_yuv.reshape(visionInfo.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, visionInfo.widthAligned)
    # img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR_NV12)

    img = cv2.imread(img_path)
    print(inference(img, show_result=True, target_shape=(260, 260)))

    # destroy streams

    streamManagerApi.DestroyAllStreams()
