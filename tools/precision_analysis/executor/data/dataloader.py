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

from utils.checker import check_loading_path


class DataLoader(object):
    def __init__(self, loading_path, *args, **kwargs):
        check_loading_path(loading_path, "loading_path")
        self.loading_path = loading_path
        self.load_dataset(*args, **kwargs)

        self.batch_size = 1

    def load_dataset(self, *args, **kwargs):
        raise NotImplementedError("Please specify a data loading method.")

    def __iter__(self):
        raise NotImplementedError("Please specify a data loading method.")

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
