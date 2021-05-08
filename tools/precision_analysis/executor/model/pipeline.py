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
import numpy as np
from PIL import Image

from StreamManagerApi import MxDataInput, StreamManagerApi, StringVector

from utils.coding_conversion import string2binary


class Pipeline:
    def __init__(self,
                 pipeline_cfg_file=None,
                 stream_name=None,
                 in_plugin_id=0,
                 out_plugin_id=None,
                 encoding_format=None,
                 keys=None,
                 parser=None,
                 shared_params=None):
        self.stream_name = string2binary(stream_name)
        self.in_plugin_id = in_plugin_id
        self.out_plugin_id = out_plugin_id
        self.key_vec = None
        self.dataInput = MxDataInput()
        self.encoding_format = encoding_format if encoding_format else "cv"
        self.parser = parser
        self.shared_params = shared_params if shared_params is not None \
            else {}
        self.set_fetch_keys(keys)

        if not os.path.exists(pipeline_cfg_file):
            raise IsADirectoryError("Given pipeline config path is invalid.")

        with open(pipeline_cfg_file, 'rb') as f:
            pipeline_str = f.read()

        self.streams = StreamManagerApi()
        ret = self.streams.InitManager()
        if ret != 0:
            raise SystemError("Failed to init Stream manager, ret=%s" %
                              str(ret))

        if not isinstance(self.encoding_format, str) or self.encoding_format\
                not in ['cv']:
            raise ValueError("Param encoding_format must be a string among "
                             "['cv'].")

        if self.parser and not hasattr(self.parser, "__call__"):
            raise ValueError("Param parser must be callable.")

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

    def parse_ret(self):
        if self.parser:
            return self.parser(self.ret, self.shared_params)
        else:
            return self.ret

    def put(self, input_b):
        self.dataInput.data = input_b
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

    def infer(self, input_numpy):
        input_bin = self.encode(input_numpy)
        self.put(input_bin)
        self.ret = self.get()
        return self.parse_ret()

    def destroy_stream(self):
        self.streams.DestroyAllStreams()

    def encode(self, input_numpy):
        if self.encoding_format == "cv":
            _, buf = cv2.imencode(".jpg", input_numpy)
            return Image.fromarray(np.uint8(buf)).tobytes()

    def __call__(self, input_numpy):
        return self.infer(input_numpy)
