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


class PipeElement(object):
    def __init__(self, name="my_element", shard_params=None):
        if isinstance(name, str):
            self.name = name
        else:
            raise ValueError("Please specify a string as the instance name.")

        if isinstance(shard_params, (type(None), dict)):
            self.shard_params = shard_params
        else:
            raise ValueError("A valid shard_params should be 'dict'.")

    def __call__(self, input_numpy):
        self.run(input_numpy)

    def run(self, input_numpy):
        raise NotImplementedError("Please specify the run method.")
