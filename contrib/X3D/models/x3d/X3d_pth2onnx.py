# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import sys
sys.path.append('slowfast')
from slowfast.models import build_model
from slowfast.utils import checkpoint as cu
from slowfast.utils.parser import load_config, parse_args
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job

def perform_x3d_pth2onnx(output_path, cfg):
    model = build_model(cfg)

    cu.load_test_checkpoint(cfg, model)
    model.eval()

    class Tmodel(nn.Module):
        def __init__(self,outer_model):
            super(Tmodel,self).__init__()
            self.outer_model = outer_model
        
        def forward(self, x): 
            x = x.unsqueeze(0)
            x = x.permute(0,2,1,3,4)
            return self.outer_model([x])

    tmodel = Tmodel(model)
    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(13, 3, 182, 182)
    torch.onnx.export(tmodel, dummy_input, output_path, input_names = input_names, dynamic_axes = dynamic_axes, output_names = output_names, opset_version=11, verbose=True)

def x3d_pth2onnx(cfg):
    output_path = cfg.X3D_PTH2ONNX.ONNX_OUTPUT_PATH
    perform_x3d_pth2onnx(output_path, cfg)

if __name__== '__main__':
    args = parse_args()
    config = load_config(args)
    config = assert_and_infer_cfg(config)
    launch_job(cfg=config, init_method=args.init_method, func=x3d_pth2onnx)
