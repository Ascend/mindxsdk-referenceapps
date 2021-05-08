#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2020 Huawei Technologies Co., Ltd

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

import os
import cv2
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import MxDataInput, StreamManagerApi, StringVector, InProtobufVector, MxProtobufIn
import numpy as np
import sys
import argparse
import mmcv
import threading
from multiprocessing import Process
from multiprocessing import Queue
import time


def preprocess_FasterRCNN_mmdet(input_image):
    # define the output file name
    one_img = mmcv.imread(os.path.join(input_image))
    two_img = one_img.copy()
    one_img = mmcv.imresize(one_img, (1216, 800))
    mean = np.array([123.675, 116.28, 103.53], np.float32)
    std = np.array([58.395, 57.12, 57.375], np.float32)
    one_img = mmcv.imnormalize(one_img, mean, std)
    one_img = one_img.transpose(2, 0, 1)
    return one_img, two_img


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--src_pic_path', type=str, default='../picture',
                        help='ground truth, having default value ./test')
    parser.add_argument('--des_pic_path', type=str, default='./output_multi/',
                        help='des box txt, having default value ./output_path/')
    return parser.parse_args(argv)


def readlabels(labelconf):
    labellist = dict()
    linenum = 0
    with open(labelconf, 'rb') as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            labellist[linenum] = line.decode().replace('\n', '')
            linenum = linenum + 1
    return labellist


class sendStreamThread(threading.Thread):
    def __init__(self, deviceId, streamName, src_dir_name, streamManagerApi, qrecv, qsend, count):
        threading.Thread.__init__(self)
        self.deviceId = deviceId
        self.streamName = streamName
        self.src_dir_name = src_dir_name
        self.streamManagerApi = streamManagerApi
        self.qrecv = qrecv
        self.qsend = qsend
        self.count = count

    def __sendStream(self, img):
        print("start sendStream")
        start_time = time.time()

        img = img.astype(np.float32)
        array_bytes = img.tobytes()
        inPluginId = 0
        dataInput = MxDataInput()
        dataInput.data = array_bytes
        key = b'appsrc1'
        protobufVec = InProtobufVector()
        visionList = MxpiDataType.MxpiVisionList()
        visionVec = visionList.visionVec.add()
        visionVec.visionInfo.format = 1
        visionVec.visionInfo.width = 1216
        visionVec.visionInfo.height = 800
        visionVec.visionInfo.widthAligned = 1216
        visionVec.visionInfo.heightAligned = 800
        visionVec.visionData.deviceId = self.deviceId
        visionVec.visionData.memType = 0
        visionVec.visionData.dataStr = dataInput.data
        visionVec.visionData.dataSize = 11673600
        protobuf = MxProtobufIn()
        protobuf.key = key
        protobuf.type = b'MxTools.MxpiVisionList'
        protobuf.protobuf = visionList.SerializeToString()
        protobufVec.push_back(protobuf)

        uniqueId = self.streamManagerApi.SendProtobuf(self.streamName, inPluginId, protobufVec)

        end_time = time.time()
        print("sendStream:", end_time - start_time)
        if uniqueId < 0:
            print("Failed to send data to stream.")
            return False
        return True

    def __getStream(self):
        start_time = time.time()
        inPluginId = 0
        keyVec = StringVector()
        keyVec.push_back(b'mxpi_modelinfer0')
        inferResult = self.streamManagerApi.GetProtobuf(self.streamName, inPluginId, keyVec)
        if inferResult.size() == 0:
            print("inferResult is null")
            exit()
        if inferResult[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (
                inferResult[0].errorCode))
            exit()

        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(inferResult[0].messageBuf)
        bbox_conf = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, np.float32)
        class_id = np.frombuffer(result.tensorPackageVec[0].tensorVec[1].dataStr, np.int64)
        bbox_conf_vec = []
        for i in range(0, len(bbox_conf), 5):
            temp = []
            for j in range(5):
                temp.append(bbox_conf[j + i])
            temp.append(class_id[i // 5])
            bbox_conf_vec.append(temp)
        bbox_conf_vec = np.array(bbox_conf_vec)
        bbox = bbox_conf_vec[:, :4]
        scores = bbox_conf_vec[:, 4]
        class_id = bbox_conf_vec[:, 5]
        order = scores.ravel().argsort()[::-1]
        order = order[:10]
        class_id = class_id[order]
        scores = scores[order]
        bbox = bbox[order, :]
        keep = np.where(scores > 0.40)[0]
        class_id = class_id[keep]
        scores = scores[keep]
        bbox = bbox[keep, :]
        end_time = time.time()
        print("recvStream:", end_time - start_time)
        return class_id, scores, bbox

    def run(self):
        q11 = Queue()
        while True:
            sendCount = 0
            while True:
                file_name = self.qrecv.get()
                if file_name == "NULL":
                    break
                else:
                    print(file_name)
                    file_path = [self.src_dir_name, "/", file_name]
                    file_path = "".join(file_path)
                    img, src_img = preprocess_FasterRCNN_mmdet(file_path)
                    if self.__sendStream(img):
                        q11.put(file_name)
                        sendCount = sendCount + 1

                if sendCount == self.count:
                    break

            for i in range(0, sendCount):
                file_name = q11.get()
                class_id, scores, bbox = self.__getStream()
                self.qsend.put((file_name, class_id, scores, bbox))

            if sendCount != self.count:
                self.qsend.put(("NULL", "", "", ""))
                break


class streamProcess(Process):
    def __init__(self, deviceId, streamName, src_dir_name, qrecv, qsend, count):
        Process.__init__(self)
        self.deviceId = deviceId
        self.streamName = streamName
        self.src_dir_name = src_dir_name
        self.qrecv = qrecv
        self.qsend = qsend
        self.count = count

    def run(self):
        streamManagerApi = StreamManagerApi()
        ret = streamManagerApi.InitManager()
        if ret != 0:
            print("Failed to init Stream manager, ret=%s" % str(ret))
            exit()

        # create streams by pipeline config file
        with open(pipeline_path, 'rb') as f:
            pipelineStr = f.read()
        ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
        if ret != 0:
            print("Failed to create Stream, ret=%s" % str(ret))
            exit()

        t1 = sendStreamThread(self.deviceId, self.streamName, self.src_dir_name, streamManagerApi, self.qrecv,
                              self.qsend, self.count)
        t1.start()
        t1.join()

        streamManagerApi.DestroyAllStreams()


class postStreamProcess(Process):
    def __init__(self, labellist, src_dir_name, res_dir_name, qrecv):
        Process.__init__(self)
        self.labellist = labellist
        self.src_dir_name = src_dir_name
        self.res_dir_name = res_dir_name
        self.qrecv = qrecv

    def __savePic(self, file_name, class_id, scores, bbox):
        file_path = [self.src_dir_name, "/", file_name]
        file_path = "".join(file_path)
        portion = os.path.splitext(file_name)
        img, src_img = preprocess_FasterRCNN_mmdet(file_path)

        img = img.astype(np.float32)
        img = img.transpose(1, 2, 0)
        for index, value in enumerate(bbox):  # y1, x1, y2, x2 -> x1 y1 x2 y2
            src_shape = src_img.shape
            dst_shape = img.shape
            dw = dst_shape[1] / src_shape[1]
            dh = dst_shape[0] / src_shape[0]
            labelname = self.labellist[int(class_id[index])]
            text = [labelname, ":", str(scores[index])]
            text = "".join(text)
            cv2.rectangle(src_img, (int(value[0] / dw), int(value[1] / dh)), (int(value[2] / dw), int(value[3] / dh)),
                          (0, 255, 0))
            font = cv2.FONT_HERSHEY_SIMPLEX
            print("result: ", text)
            cv2.putText(src_img, text, (int(value[0] / dw), int((value[1] / dh + 30))), font, 1, (0, 0, 255), 2)
        cv2.imwrite(self.res_dir_name + portion[0] + '_res' + portion[1], src_img)

    def run(self):
        while True:
            file_name, class_id, scores, bbox = self.qrecv.get()
            if file_name == "NULL":
                print("poss stream process finish")
                break
            else:
                self.__savePic(file_name, class_id, scores, bbox)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    pipeline_path = "./EasyStream_protobuf.pipeline"
    labelconf = "../models/cascadercnn/faster_rcnn_coco.names"
    labellist = readlabels(labelconf)
    src_dir_name = args.src_pic_path
    res_dir_name = args.des_pic_path

    q0 = Queue()
    q1 = Queue()

    p1 = streamProcess(0, b'detection0', src_dir_name, q0, q1, 4)
    p2 = streamProcess(1, b'detection1', src_dir_name, q0, q1, 4)
    p3 = streamProcess(2, b'detection2', src_dir_name, q0, q1, 4)
    p4 = streamProcess(3, b'detection3', src_dir_name, q0, q1, 4)

    p5 = postStreamProcess(labellist, src_dir_name, res_dir_name, q1)
    p6 = postStreamProcess(labellist, src_dir_name, res_dir_name, q1)
    p7 = postStreamProcess(labellist, src_dir_name, res_dir_name, q1)
    p8 = postStreamProcess(labellist, src_dir_name, res_dir_name, q1)

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()

    start_time = time.time()
    # Inputs data to a specified stream based on streamName.
    file_list = os.listdir(src_dir_name)
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)
    for file_name in file_list:
        if file_name.endswith(".JPG") or file_name.endswith(".jpg"):
            q0.put(file_name)

    q0.put("NULL")
    q0.put("NULL")
    q0.put("NULL")
    q0.put("NULL")

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()

    end_time = time.time()
    print('total:\n', end_time - start_time)
