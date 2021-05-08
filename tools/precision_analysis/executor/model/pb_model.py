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

import cv2
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile

from executor.model.base import ModelExecutor


class PBModel(ModelExecutor):
    def __init__(self, model_path, input_name, output_name):
        super(PBModel, self).__init__(model_path)
        if not isinstance(input_name, list):
            raise ValueError("Param input_name should be a list.")

        if not isinstance(output_name, list):
            raise ValueError("Param output_name should be a list.")

        if isinstance(input_name, str):
            self.input_name = [input_name]
            self.input_cnt = 1
        elif isinstance(input_name, list):
            self.input_cnt = len(input_name)
        else:
            raise ValueError("Param input_name should be either str or list.")

        if isinstance(output_name, str):
            self.output_name = [output_name]
            self.output_cnt = 1
        elif isinstance(output_name, list):
            self.output_cnt = len(output_name)
        else:
            raise ValueError("Param output_name should be either str or list.")

        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def build_model(self, model_path):
        """
        :param model_path:
        """
        config = tf.ConfigProto()
        self.sess = tf.Session(config=config)
        with gfile.FastGFile(model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def, name="")

    def get_input_tensor(self):
        self.input_tensor = []
        for name in self.input_name:
            tensor = tf.get_default_graph().get_tensor_by_name(name)
            self.input_tensor.append(tensor)

    def get_output_tensor(self):
        self.output_tensor = []
        for name in self.output_name:
            tensor = tf.get_default_graph().get_tensor_by_name(name)
            self.output_tensor.append(tensor)

    def get_input_feed(self, image_numpy):
        if isinstance(image_numpy, np.ndarray) and self.input_cnt == 1:
            image_numpy = [image_numpy]

        input_feed = {}
        for tensor, input_data in zip(self.input_tensor, image_numpy):
            input_feed[tensor] = input_data
        return input_feed

    def forward(self, input_numpy):
        input_feed = self.get_input_feed(input_numpy)
        return self.sess.run(self.output_tensor, feed_dict=input_feed)


if __name__ == "__main__":
    input_name = ""
    output_name = [
        "detector/yolo-v4-tiny/Conv_17/BiasAdd:0",
        "detector/yolo-v4-tiny/Conv_20/BiasAdd:0"
    ]
    model_path = ""
    img_path = ""
    img = cv2.imread(img_path)
    model = PBModel(model_path, input_name, output_name)
    out = model(img)
    print(out)
