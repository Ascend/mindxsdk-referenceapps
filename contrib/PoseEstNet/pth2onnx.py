# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

import argparse
import os

import torch
import torch.backends.cudnn as cudnn

from config import cfg
from config import update_config

import models


def main():
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args = parser.parse_args()
    update_config(cfg, args)

    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    net = eval('models.pose_hrnet.get_pose_net')(
        cfg, is_train=False
    )

    net.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
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
