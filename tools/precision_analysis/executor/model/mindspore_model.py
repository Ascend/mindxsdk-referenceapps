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

import numpy as np

from mindspore import Model, load_checkpoint, load_param_into_net, Tensor
from mindspore.nn import Cell

from executor.model.base import ModelExecutor


class MSModel(ModelExecutor):
    def build_model(self, model_path):
        """
        :param model_path:
        """
        network = self.build_graph()
        if not isinstance(network, Cell):
            raise ValueError("Please return a Cell instance.")

        self.model = Model(network)
        param_dict = load_checkpoint(model_path)
        load_param_into_net(network, param_dict)

    def build_graph(self, *args, **kwargs) -> Cell:
        raise NotImplementedError("Please specify a graph building method.")

    def forward(self, input_numpy):
        if isinstance(input_numpy, np.ndarray):
            input_tensor = [input_numpy]

        else:
            input_tensor = input_numpy

        input_tensor = [Tensor(tensor) for tensor in input_tensor]
        output = self.model.predict(*input_tensor)
        return output
