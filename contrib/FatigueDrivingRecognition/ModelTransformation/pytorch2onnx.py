# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import onnx
import os
import argparse
from models.pfld import PFLDInference
import torch
import onnxsim
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pytorch2onnx')
    parser.add_argument(
        '--torch_model',
        default="./checkpoint/v2/v2.pth")
    parser.add_argument('--onnx_model', default="./pfld_106.onnx")
    args = parser.parse_args()
    # load pytorch checkpoint
    print("=====> load pytorch checkpoint...")
    plfd_backbone = PFLDInference()
    plfd_backbone.eval()
    plfd_backbone.load_state_dict(torch.load(args.torch_model, map_location=torch.device('cpu'))['plfd_backbone'])
    print("PFLD bachbone:", plfd_backbone)
    # convert pytorch model to onnx
    print("=====> convert pytorch model to onnx...")
    dummy_input = torch.randn(1, 3, 112, 112)
    input_names = ["input_1"]
    output_names = ["output_1"]
    
    
    torch.onnx.export(
        plfd_backbone,
        dummy_input,
        args.onnx_model,
        verbose=True,
        opset_version = 11,
        input_names=input_names,
        output_names=output_names)
    
    # check onnx model
    print("====> check onnx model...")
    model = onnx.load(args.onnx_model)
    onnx.checker.check_model(model)