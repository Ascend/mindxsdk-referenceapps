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

from test.build import TEST_MAP
from utils.arguments import get_args
from utils import constants


def main():
    args = get_args()
    mode_str = args.mode
    data_loading_path = args.data_loading_path
    label_loading_path = args.label_loading_path
    pipeline_cfg_path = args.pipeline_cfg_path
    stream_name = args.stream_name

    cfg = {"data_loading_path": data_loading_path,
           "label_loading_path": label_loading_path,
           "pipeline_cfg_path": pipeline_cfg_path,
           "stream_name": stream_name}

    mode_list = mode_str.split(".")
    if len(mode_list) <= 1:
        raise NotImplementedError("Only test mode is supported.")

    elif len(mode_list) != 3:
        raise ValueError(f"Please offer a format like "
                         f"'test.{constants.ModelName.CRNN.value}"
                         f".{constants.UnitName.PIPELINE.value}'.")

    mode = mode_list[0].lower()
    model_name = mode_list[1].lower()
    testing_unit = mode_list[2].lower()

    if mode != "test":
        raise ValueError("Only test mode is supported.")

    if model_name not in [
            constants.ModelName.CRNN.value, constants.ModelName.SMF.value
    ]:
        raise ValueError(f"We can only offer demos with respect to models ["
                         f"{constants.ModelName.CRNN.value}, "
                         f"{constants.ModelName.SMF.value}].")

    if testing_unit not in [
            constants.UnitName.PIPELINE.value,
            constants.UnitName.INFERENCE.value,
            constants.UnitName.EVALUATION.value
    ]:
        raise ValueError(f"Only 3 unit, {constants.UnitName.PIPELINE.value} "
                         f"{constants.UnitName.INFERENCE.value} and "
                         f"{constants.UnitName.EVALUATION.value}, "
                         f"can be tested.")

    exec_func = TEST_MAP.get(model_name).get(testing_unit)
    exec_func(cfg)


if __name__ == '__main__':
    main()
