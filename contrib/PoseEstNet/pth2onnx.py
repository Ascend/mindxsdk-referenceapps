#!/usr/bin/env python
# coding=utf-8

"""
Copyright(C) Huawei Technologies Co.,Ltd. 2012-2022 All rights reserved.

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

import argparse
import os

import torch
import torch.backends.cudnn as cudnn

import _init_paths
from config import cfg
from config import update_config

import models


def main():
    parser = argparse.ArgumentParser(description='Transform model')
    parser.add_argument('--cfg',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        type=str,
                        default='')

    parser.add_argument('--logDir',
                        type=str,
                        default='')

    parser.add_argument('--dataDir',
                        type=str,
                        default='')

    args = parser.parse_args()
    update_config(cfg, args)

    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    net = eval('models.pose_hrnet.get_pose_net')(
        cfg, is_train=False
    )

    net.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), map_location="cpu", strict=False)
    net.eval()

    device = torch.device("cpu")
    dummy_input = torch.randn(1, 3, 256, 256, device=device)

    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    export_onnx_file = "PoseEstNet.onnx"

    torch.onnx.export(net, dummy_input, export_onnx_file, input_names=input_names, dynamic_axes=dynamic_axes,
                      output_names=output_names, opset_version=11, verbose=True)


if __name__ == '__main__':
    main()
