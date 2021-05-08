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

import onnxruntime
import numpy as np

from executor.model.base import ModelExecutor


class ONNXModel(ModelExecutor):
    def build_model(self, model_path):
        """
        :param model_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self):
        self.output_name = []
        for node in self.onnx_session.get_outputs():
            self.output_name.append(node.name)

        self.output_cnt = len(self.output_name)

    def get_input_name(self):
        self.input_name = []
        for node in self.onnx_session.get_inputs():
            self.input_name.append(node.name)

        self.input_cnt = len(self.input_name)

    def get_input_feed(self, image_numpy):
        if isinstance(image_numpy, np.ndarray) and self.input_cnt == 1:
            image_numpy = [image_numpy]

        input_feed = {}
        for name, input_data in zip(self.input_name, image_numpy):
            input_feed[name] = input_data
        return input_feed

    def forward(self, input_numpy):
        input_feed = self.get_input_feed(input_numpy)
        output = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return output
