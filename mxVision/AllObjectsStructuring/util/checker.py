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


def check_loading_path(loading_path, path_name, suffix_set=()):
    if not isinstance(suffix_set, (list, tuple)):
        suffix_set = [suffix_set]

    for item in suffix_set:
        if not isinstance(item, str):
            raise TypeError("Wrong suffix type.")

    if not isinstance(loading_path, str):
        raise ValueError(f"please specify a string as the {path_name}")

    _, suffix = os.path.splitext(loading_path)
    if suffix not in suffix_set:
        raise ValueError(f"Please specify a {path_name} with suffix '.index'.")

    if not os.path.exists(loading_path):
        raise NotADirectoryError(f"Given {path_name} does not exist.")


def check_saving_path(saving_path, path_name, suffix_set=()):
    if not isinstance(suffix_set, (list, tuple)):
        suffix_set = [suffix_set]

    for item in suffix_set:
        if not isinstance(item, str):
            raise TypeError("Wrong suffix type.")

    if not isinstance(saving_path, str):
        raise ValueError(f"please assign a string as the {path_name}")

    root_dir, file_name = os.path.split(saving_path)
    root_dir = "./" if root_dir == "" else root_dir

    os.makedirs(root_dir, exist_ok=True)

    _, suffix = os.path.splitext(file_name)

    if suffix not in suffix_set:
        raise ValueError(f"Please assign suffix '.index' to {path_name}.")
