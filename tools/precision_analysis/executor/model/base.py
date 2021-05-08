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

from utils.checker import check_loading_path, check_model_input


class ModelExecutor(object):
    def __init__(self, model_path):
        self.input_name = None
        self.output_name = None
        self.input_cnt = 1
        self.output_cnt = 1
        check_loading_path(model_path, "model_path")
        self.build_model(model_path)

    def build_model(self, model_path):
        raise NotImplementedError(f"Please specify a graph building method.")

    def infer(self, input_numpy):
        raise NotImplementedError(f"Please specify a model infer method.")

    def __call__(self, input_numpy):
        check_model_input(input_numpy)
        return self.infer(input_numpy)
