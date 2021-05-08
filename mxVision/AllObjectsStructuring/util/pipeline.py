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
import datetime
from time import sleep
from google.protobuf import json_format

from util.yuv import yuv2bgr
from StreamManagerApi import MxDataInput, StreamManagerApi, StringVector
import MxpiDataType_pb2 as MxpiDataType
import MxpiAllObjectsStructuringDataType_pb2 as WebDisplayDataType


def binary2string(string):
    if isinstance(string, str):
        return string

    return string.decode()


def string2binary(string):
    if isinstance(string, str):
        return string.encode()

    return string


class Pipeline:
    key_name2data_struc = {
        'face_attribute': 'MxpiAttributeList',
        'face_feature': 'MxpiFeatureVectorList',
        'mxpi_facealignment0': 'MxpiVisionList',  # 获取人脸图片
        'mxpi_parallel2serial2': 'MxpiVisionList',  # 获取行人车辆图片
        'ReservedFrameInfo': 'MxpiFrameInfo',
        'motor_attr': 'MxpiAttributeList',
        'car_plate': 'MxpiAttributeList',
        'pedestrian_attribute': 'MxpiAttributeList',
        'pedestrian_reid': 'MxpiFeatureVectorList',
        'vision': 'MxpiVisionList',
        'object': 'MxpiObjectList',
        'mxpi_framealign0': 'MxpiWebDisplayDataList'
    }
    clr_cvt_map = {
        "face": "COLOR_YUV2BGR_NV12",  # "COLOR_YUV2BGR_I420",
        "motor-vehicle": "COLOR_YUV2BGR_NV12",
        "person": "COLOR_YUV2BGR_NV12"
    }

    def __init__(self,
                 pipeline_cfg_file=None,
                 stream_name=None,
                 in_plugin_id=None,
                 out_plugin_id=None,
                 out_stream_plugin_id=None,
                 keys=None,
                 stream_bbox_key=None):
        self.stream_name = string2binary(stream_name)
        self.in_plugin_id = in_plugin_id
        self.out_plugin_id = out_plugin_id
        self.out_stream_plugin_id = out_stream_plugin_id
        self.key_vec = None
        self.stream_key_vec = None
        self.bbox_key_vec = None
        self.dataInput = MxDataInput()
        self.set_fetch_keys(keys)
        self.set_stream_bbox_keys(stream_bbox_key)
        self.infer_result_has_errorCode = False
        self.stream_bbox_data_has_errorCode = False

        if not os.path.exists(pipeline_cfg_file):
            raise IsADirectoryError("Given pipeline config path is invalid.")

        with open(pipeline_cfg_file, 'rb') as f:
            pipeline_str = f.read()

        self.streams = StreamManagerApi()
        ret = self.streams.InitManager()
        if ret != 0:
            raise SystemError("Failed to init Stream manager, ret=%s" %
                              str(ret))

        ret = self.streams.CreateMultipleStreams(pipeline_str)
        if ret != 0:
            raise IOError("Failed to create Stream, ret=%s" % str(ret))

    def set_fetch_keys(self, keys):
        if not keys:
            return
        if isinstance(keys, (list, tuple)):
            self.key_vec = StringVector()
            for key in keys:
                self.key_vec.push_back(string2binary(key))

    def set_stream_bbox_keys(self, stream_bbox_key):
        if not stream_bbox_key:
            return
        if isinstance(stream_bbox_key, (list, tuple)):
            self.stream_key_vec = StringVector()
            for key in stream_bbox_key:
                self.stream_key_vec.push_back(string2binary(key))

    def get_single_stream_ret(self, save_fig=False, base64_enc=False):
        ret_dict = {
            "stream_name": binary2string(self.stream_name),
            "out_plugin_id": self.out_plugin_id,
            "channel_id": None,
            "frame_id": None,
            "object_name": None,
            "image": None,
            "attribute": None,
            "feature_vector": None
        }
        while True:
            infer_result = self.streams.GetProtobuf(self.stream_name,
                                                    self.out_plugin_id,
                                                    self.key_vec)
            if infer_result.size() == 0:
                sleep(0.1)
                continue

            for item in infer_result:
                if item.errorCode != 0:
                    self.infer_result_has_errorCode = True
                    continue

                self.parse_item(item, ret_dict)

            if self.infer_result_has_errorCode:
                self.infer_result_has_errorCode = False
                sleep(0.1)
                continue

            self.generate_fig(ret_dict,
                              save_fig=save_fig,
                              base64_enc=base64_enc)

            if ret_dict["object_name"]:
                ret_dict["object_index"] = "_".join([
                    ret_dict.get("stream_name"),
                    str(ret_dict.get("channel_id")),
                    str(ret_dict.get("frame_id")),
                    ret_dict.get("object_name")
                ])
            else:
                continue
            return ret_dict

    def get_stream_bbox_data(self):
        ret_dict = {
            "stream_name": binary2string(self.stream_name),
            "out_plugin_id": self.out_stream_plugin_id,
            "channel_id": None,
            "frame_id": None,
            "web_display_data_serialized": None,
            "web_display_data_dict": None,
        }
        while True:
            sleep(0.01)
            infer_result = self.streams.GetProtobuf(self.stream_name,
                                                    self.out_stream_plugin_id,
                                                    self.stream_key_vec)

            if infer_result.size() == 0:
                sleep(0.1)
                continue

            item = infer_result[0]
            if item.errorCode != 0:
                continue
            self.parse_item(item, ret_dict)

            return ret_dict

    def generate_fig(self, ret_dict, save_fig=False, base64_enc=False):
        image = ret_dict.get("image") if ret_dict.get("image") is not None \
            else {}
        image_b = image.get("image_b")
        height = image.get("height")
        width = image.get("width")
        if not image_b:
            return

        ret_dict["image"] = None
        save_dir = None
        if save_fig:
            object_name = ret_dict.get("object_name")
            channel_id = ret_dict.get("channel_id")
            dir_name = object_name + "/" + str(channel_id)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            file_name = f"{datetime.datetime.now().timestamp()}.jpg"
            save_dir = os.path.join(dir_name, file_name)

        object_name = ret_dict.get("object_name")
        clr_cvt_mtd = self.clr_cvt_map.get(object_name)
        if not clr_cvt_mtd:
            raise ValueError(f"Cannot find the color convert method with "
                             f"respect to object {object_name}")

        cvt_ret = yuv2bgr(image_b,
                          height,
                          width,
                          clr_cvt_mtd=clr_cvt_mtd,
                          output_path=save_dir,
                          base64_enc=base64_enc)
        if cvt_ret:
            ret_dict['image_encoded'] = cvt_ret

    def parse_item(self, item, ret_dict: dict):
        item_key = binary2string(item.messageName)
        item_value = item.messageBuf
        data_struc = self.key_name2data_struc[item_key]
        if data_struc == 'MxpiWebDisplayDataList':
            data_parser = getattr(WebDisplayDataType, data_struc)()
        else:
            data_parser = getattr(MxpiDataType, data_struc)()
        data_parser.ParseFromString(item_value)

        if data_struc == 'MxpiVisionList':
            # Todo 目标类别识别方法需要优化
            if item_key == "mxpi_parallel2serial2":
                ret_dict["object_name"] = "face"

            ret_dict["image"] = \
                {"image_b": data_parser.visionVec[0].visionData.dataStr,
                 "height": data_parser.visionVec[0].visionInfo.heightAligned,
                 "width": data_parser.visionVec[0].visionInfo.widthAligned}

        elif data_struc == 'MxpiFeatureVectorList':
            result = json_format.MessageToDict(data_parser)
            ret_dict["feature_vector"] = result['featureVec'][0][
                'featureValues']

        elif data_struc == 'MxpiAttributeList':
            result = json_format.MessageToDict(data_parser)
            if ret_dict["attribute"] is None:
                ret_dict["attribute"] = result['attributeVec']
            else:
                for item in result['attributeVec']:
                    ret_dict["attribute"].append(item)

        elif data_struc == 'MxpiFrameInfo':
            ret_dict["channel_id"] = data_parser.channelId
            ret_dict["frame_id"] = data_parser.frameId

        elif data_struc == 'MxpiObjectList':
            result = json_format.MessageToDict(data_parser)
            ret_dict["object"] = result
            ret_dict["object_name"] = result['objectVec'][0]['classVec'][0][
                'className']

        elif data_struc == 'MxpiWebDisplayDataList':
            result = data_parser.webDisplayDataVec[0].SerializeToString()
            result_dict = json_format.MessageToDict(data_parser.webDisplayDataVec[0])
            ret_dict["channel_id"] = data_parser.webDisplayDataVec[0].channel_id
            ret_dict["frame_id"] = data_parser.webDisplayDataVec[0].frame_index
            ret_dict["web_display_data_serialized"] = result
            ret_dict["web_display_data_dict"] = result_dict

    def put(self, image_b):
        """
        send data only
        """
        self.dataInput.data = image_b
        self.out_plugin_id = self.streams.SendDataWithUniqueId(
            self.stream_name, self.in_plugin_id, self.dataInput)

    def get(self):
        inferResult = self.streams.GetResultWithUniqueId(
            self.stream_name, self.out_plugin_id, 3000)
        if inferResult.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, "
                  "errorMsg=%s" %
                  (inferResult.errorCode, inferResult.data.decode()))

        else:
            return inferResult.data.decode()

    def infer(self, image_bin):
        self.put(image_bin)
        return self.get()

    def destory_stream(self):
        self.streams.DestroyAllStreams()


if __name__ == '__main__':
    pipeline_cfg_file_path = "../pipeline/AllObjectsStructuring" \
                             ".pipeline"
    stream_name_str = "detection"
    plugin_id = 0
    desired_keys = [
        b'face_attribute', b'face_feature', b'mxpi_parallel2serial2',
        b'motor_attr', b'car_plate', b'ReservedFrameInfo',
        b'pedestrian_attribute', b'pedestrian_reid', b'vision', b'object'
    ]
    pipeline = Pipeline(pipeline_cfg_file=pipeline_cfg_file_path,
                        stream_name=stream_name_str,
                        out_plugin_id=plugin_id,
                        keys=desired_keys)
    while True:
        buffer = pipeline.get_single_stream_ret(save_fig=True, base64_enc=True)
        print(buffer)
